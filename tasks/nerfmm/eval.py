import sys
import os
import argparse
from pathlib import Path
import logging

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import imageio
import lpips as lpips_lib

sys.path.append(os.path.join(sys.path[0], '../..'))

from dataloader.with_colmap import DataLoaderWithCOLMAP
from utils.training_utils import set_randomness, mse2psnr, load_ckpt_to_net
from utils.comp_ray_dir import comp_ray_dir_cam_fxfy
from third_party import pytorch_ssim
from utils.align_traj import align_scale_c2b_use_a2b
from models.nerf_models import OfficialNerf
from models.intrinsics import LearnFocal
from models.poses import LearnPose
from tasks.nerfmm.train import model_render_image, eval_one_epoch_traj


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--multi_gpu',  default=False, action='store_true')
    parser.add_argument('--base_dir', type=str, default='./data_dir/nerfmm_release_data',
                        help='folder contains various scenes')
    parser.add_argument('--scene_name', type=str, default='LLFF/fern')
    parser.add_argument('--use_ndc', type=bool, default=True)

    parser.add_argument('--learn_focal', default=False, type=bool)
    parser.add_argument('--fx_only', default=False, type=eval, choices=[True, False])
    parser.add_argument('--focal_order', default=2, type=int)

    parser.add_argument('--learn_R', default=False, type=bool)
    parser.add_argument('--learn_t', default=False, type=bool)

    parser.add_argument('--init_pose_from', default='none', type=str, choices=['colmap', 'none'],
                        help='set it to "colmap" if the nerfmm is trained from colmap pose initialisation.')
    parser.add_argument('--init_focal_from', default='none', type=str, choices=['colmap', 'none'],
                        help='set it to "colmap" if the nerfmm is trained from colmap focal initialisation.')

    parser.add_argument('--resize_ratio', type=int, default=4, help='lower the image resolution with this ratio')
    parser.add_argument('--num_rows_eval_img', type=int, default=10, help='split a high res image to rows in eval')
    parser.add_argument('--hidden_dims', type=int, default=128, help='network hidden unit dimensions')
    parser.add_argument('--train_rand_rows', type=int, default=32, help='rand sample these rows to train')
    parser.add_argument('--train_rand_cols', type=int, default=32, help='rand sample these cols to train')
    parser.add_argument('--num_sample', type=int, default=128, help='number samples along a ray')

    parser.add_argument('--pos_enc_levels', type=int, default=10, help='number of freqs for positional encoding')
    parser.add_argument('--pos_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--use_dir_enc', type=bool, default=True, help='use pos enc for view dir?')
    parser.add_argument('--dir_enc_levels', type=int, default=4, help='number of freqs for positional encoding')
    parser.add_argument('--dir_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--opt_pose_epoch', default=1000, type=int)
    parser.add_argument('--opt_eval_lr', default=0.001, type=float)
    parser.add_argument('--opt_eval_R', default=True, type=bool)
    parser.add_argument('--opt_eval_t', default=True, type=bool)
    parser.add_argument('--eval_pose_milestones', default=list(range(0, 1000, 200)), type=int, nargs='+',
                        help='learning rate schedule milestones')
    parser.add_argument('--eval_pose_lr_gamma', type=float, default=0.5, help="learning rate milestones gamma")

    parser.add_argument('--type_to_eval', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--train_img_num', type=int, default=-1, help='num of images to train')
    parser.add_argument('--train_skip', type=int, default=1, help='skip every this number of imgs')
    parser.add_argument('--eval_img_num', type=int, default=-1, help='num of images to eval')
    parser.add_argument('--eval_skip', type=int, default=1, help='skip every this number of imgs')

    parser.add_argument('--rand_seed', type=int, default=17)
    parser.add_argument('--true_rand', type=bool, default=False)

    parser.add_argument('--ckpt_dir', type=str, default='')
    return parser.parse_args()


def eval_one_epoch_img(scene_eval, model, focal_net, eval_pose_param_net, my_devices, args, logger):
    model.eval()
    focal_net.eval()
    eval_pose_param_net.eval()

    # init lpips loss.
    lpips_vgg_fn = lpips_lib.LPIPS(net='vgg').to(my_devices)

    # load learned intrinsic and extrinsic
    fxfy = focal_net(0)  # (2, )

    ray_dir_cam = comp_ray_dir_cam_fxfy(scene_eval.H, scene_eval.W, fxfy[0], fxfy[1])
    t_vals = torch.linspace(scene_eval.near, scene_eval.far, args.num_sample, device=my_devices)  # (N_sample,) sample position
    N_img = scene_eval.N_imgs

    rendered_img_list = []
    rendered_depth_list = []

    eval_mse_list = []
    eval_psnr_list = []
    eval_ssim_list = []
    eval_lpips_list = []

    for i in tqdm(range(N_img)):
        img = scene_eval.imgs[i].to(my_devices)

        # if eval training set, this contains exact learned poses, if eval val set, this contains optimised eval poses.
        c2w = eval_pose_param_net(i)  # (4, 4)

        # split an image to rows when the input image resolution is high
        rays_dir_cam_split_rows = ray_dir_cam.split(args.num_rows_eval_img, dim=0)
        rendered_img = []
        rendered_depth = []
        for rays_dir_rows in rays_dir_cam_split_rows:
            render_result = model_render_image(c2w, rays_dir_rows, t_vals, scene_eval.near, scene_eval.far,
                                               scene_eval.H, scene_eval.W, fxfy, model, False,
                                               0.0, args, rgb_act_fn=torch.sigmoid)
            rgb_rendered_rows = render_result['rgb']  # (num_rows_eval_img, W, 3)
            depth_map = render_result['depth_map']  # (num_rows_eval_img, W)

            rendered_img.append(rgb_rendered_rows)
            rendered_depth.append(depth_map)

        # combine rows to an image
        rendered_img = torch.cat(rendered_img, dim=0)  # (H, W, 3)
        rendered_depth = torch.cat(rendered_depth, dim=0)  # (H, W)

        # mse for the entire image
        mse = F.mse_loss(rendered_img, img).item()
        psnr = mse2psnr(mse)
        ssim = pytorch_ssim.ssim(rendered_img.permute(2, 0, 1).unsqueeze(0), img.permute(2, 0, 1).unsqueeze(0)).item()
        lpips_loss = lpips_vgg_fn(rendered_img.permute(2, 0, 1).unsqueeze(0).contiguous(),
                                  img.permute(2, 0, 1).unsqueeze(0).contiguous(), normalize=True).item()

        eval_mse_list.append(mse)
        eval_psnr_list.append(psnr)
        eval_ssim_list.append(ssim)
        eval_lpips_list.append(lpips_loss)

        tqdm.write('{0:4d} img: PSNR: {1:.2f}, SSIM: {2:.2f}, LPIPS {3:.2f}'.format(i, psnr, ssim, lpips_loss))
        logger.info('{0:4d} img: PSNR: {1:.2f}, SSIM: {2:.2f}, LPIPS {3:.2f}'.format(i, psnr, ssim, lpips_loss))

        # for vis
        rendered_img_list.append(rendered_img)
        rendered_depth_list.append(rendered_depth)

    mean_mse = np.mean(eval_mse_list)
    mean_psnr = np.mean(eval_psnr_list)
    mean_ssim = np.mean(eval_ssim_list)
    mean_lpips = np.mean(eval_lpips_list)
    print('--------------------------')
    print('Mean MSE: {0:.2f}, PSNR: {1:.2f}, SSIM: {2:.2f}, LPIPS {3:.2f}'.format(mean_mse, mean_psnr,
                                                                                  mean_ssim, mean_lpips))
    logger.info('--------------------------')
    logger.info('Mean MSE: {0:.2f}, PSNR: {1:.2f}, SSIM: {2:.2f}, LPIPS {3:.2f}'.format(mean_mse, mean_psnr,
                                                                                        mean_ssim, mean_lpips))

    rendered_img_list = torch.stack(rendered_img_list)  # (N, H, W, 3)
    rendered_depth_list = torch.stack(rendered_depth_list)  # (N, H, W, 3)

    result = {
        'imgs': rendered_img_list,
        'depths': rendered_depth_list,
    }
    return result


def opt_eval_pose_one_epoch(model, focal_net, eval_pose_param_net, scene_eval, optimizer_eval_pose, my_devices):
    model.eval()
    focal_net.eval()
    eval_pose_param_net.train()

    t_vals = torch.linspace(scene_eval.near, scene_eval.far, args.num_sample,
                            device=my_devices)  # (N_sample,) sample position
    N_img, H, W = scene_eval.N_imgs, scene_eval.H, scene_eval.W

    L2_loss_epoch = []

    for i in range(N_img):
        fxfy = focal_net(0)
        ray_dir_cam = comp_ray_dir_cam_fxfy(H, W, fxfy[0], fxfy[1])
        img = scene_eval.imgs[i].to(my_devices)  # (H, W, 4)
        c2w = eval_pose_param_net(i)  # (4, 4)

        # sample pixel on an image and their rays for training.
        r_id = torch.randperm(H, device=my_devices)[:args.train_rand_rows]  # (N_select_rows)
        c_id = torch.randperm(W, device=my_devices)[:args.train_rand_cols]  # (N_select_cols)
        ray_selected_cam = ray_dir_cam[r_id][:, c_id]  # (N_select_rows, N_select_cols, 3)
        img_selected = img[r_id][:, c_id]  # (N_select_rows, N_select_cols, 3)

        # render an image using selected rays, pose, sample intervals, and the network
        render_result = model_render_image(c2w, ray_selected_cam, t_vals, scene_eval.near, scene_eval.far,
                                           scene_eval.H, scene_eval.W, fxfy,
                                           model, True, 0.0, args, torch.sigmoid)  # (N_select_rows, N_select_cols, 3)
        rgb_rendered = render_result['rgb']  # (N_select_rows, N_select_cols, 3)

        L2_loss = F.mse_loss(rgb_rendered, img_selected)  # loss for one image
        L2_loss.backward()
        optimizer_eval_pose.step()
        optimizer_eval_pose.zero_grad()

        L2_loss_epoch.append(L2_loss.item())

    L2_loss_mean = np.mean(L2_loss_epoch)
    mean_losses = {
        'L2': L2_loss_mean,
    }

    return mean_losses


def main(args):
    my_devices = torch.device('cuda:' + str(args.gpu_id))

    '''Create Folders'''
    test_dir = Path(os.path.join(args.ckpt_dir, 'render_' + args.type_to_eval))
    img_out_dir = Path(os.path.join(test_dir, 'img_out'))
    depth_out_dir = Path(os.path.join(test_dir, 'depth_out'))
    video_out_dir = Path(os.path.join(test_dir, 'video_out'))
    eval_pose_out_dir = Path(os.path.join(test_dir, 'eval_pose_out'))
    test_dir.mkdir(parents=True, exist_ok=True)
    img_out_dir.mkdir(parents=True, exist_ok=True)
    depth_out_dir.mkdir(parents=True, exist_ok=True)
    video_out_dir.mkdir(parents=True, exist_ok=True)
    eval_pose_out_dir.mkdir(parents=True, exist_ok=True)

    '''LOG'''
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    file_handler = logging.FileHandler(os.path.join(eval_pose_out_dir, 'log.txt'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.info(args)

    '''Summary Writer'''
    writer = SummaryWriter(log_dir=str(eval_pose_out_dir))

    '''Load data'''
    scene_train = DataLoaderWithCOLMAP(base_dir=args.base_dir,
                                       scene_name=args.scene_name,
                                       data_type='train',
                                       res_ratio=args.resize_ratio,
                                       num_img_to_load=args.train_img_num,
                                       skip=args.train_skip,
                                       use_ndc=args.use_ndc,
                                       load_img=args.type_to_eval == 'train')  # only load imgs if eval train set.


    print('Intrinsic: H: {0:4d}, W: {1:4d}, GT focal {2:.2f}.'.format(scene_train.H, scene_train.W, scene_train.focal))

    if args.type_to_eval == 'train':
        scene_eval = scene_train
    else:
        scene_eval = DataLoaderWithCOLMAP(base_dir=args.base_dir,
                                          scene_name=args.scene_name,
                                          data_type='val',
                                          res_ratio=args.resize_ratio,
                                          num_img_to_load=args.eval_img_num,
                                          skip=args.eval_skip,
                                          use_ndc=args.use_ndc)

    '''Model Loading'''
    pos_enc_in_dims = (2 * args.pos_enc_levels + int(args.pos_enc_inc_in)) * 3  # (2L + 0 or 1) * 3
    if args.use_dir_enc:
        dir_enc_in_dims = (2 * args.dir_enc_levels + int(args.dir_enc_inc_in)) * 3  # (2L + 0 or 1) * 3
    else:
        dir_enc_in_dims = 0

    model = OfficialNerf(pos_enc_in_dims, dir_enc_in_dims, args.hidden_dims)
    if args.multi_gpu:
        model = torch.nn.DataParallel(model).to(device=my_devices)
    else:
        model = model.to(device=my_devices)
    model = load_ckpt_to_net(os.path.join(args.ckpt_dir, 'latest_nerf.pth'), model, map_location=my_devices)

    if args.init_focal_from == 'colmap':
        focal_net = LearnFocal(scene_train.H, scene_train.W, args.learn_focal, args.fx_only, order=args.focal_order, init_focal=scene_train.focal)
    else:
        focal_net = LearnFocal(scene_train.H, scene_train.W, args.learn_focal, args.fx_only, order=args.focal_order)
    if args.multi_gpu:
        focal_net = torch.nn.DataParallel(focal_net).to(device=my_devices)
    else:
        focal_net = focal_net.to(device=my_devices)
    # load learned focal if we did not init focal with something
    if args.init_focal_from == 'none':
        focal_net = load_ckpt_to_net(os.path.join(args.ckpt_dir, 'latest_focal.pth'), focal_net, map_location=my_devices)
    fxfy = focal_net(0)
    if 'blender/' in args.scene_name:
        print('GT: fx {0:.2f} fy {1:.2f}, learned: fx {2:.2f}, fy {3:.2f}, COLMAP: {4:.2f}'.format(
            scene_train.gt_fx, scene_train.gt_fy, fxfy[0].item(), fxfy[1].item(), scene_train.focal))
    else:
        print('COLMAP: {0:.2f}, learned: fx {1:.2f}, fy {2:.2f}, '.format(
            scene_train.focal, fxfy[0].item(), fxfy[1].item()))

    if args.init_pose_from == 'colmap':
        learned_pose_param_net = LearnPose(scene_train.N_imgs, args.learn_R, args.learn_t, scene_train.c2ws)
    else:
        learned_pose_param_net = LearnPose(scene_train.N_imgs, args.learn_R, args.learn_t, None)
    if args.multi_gpu:
        learned_pose_param_net = torch.nn.DataParallel(learned_pose_param_net).to(device=my_devices)
    else:
        learned_pose_param_net = learned_pose_param_net.to(device=my_devices)
    learned_pose_param_net = load_ckpt_to_net(os.path.join(args.ckpt_dir, 'latest_pose.pth'), learned_pose_param_net,
                                              map_location=my_devices)

    # We optimise poses for validation images while freezing learned focal length and trained nerf model.
    # This step is only required when we compute evaluation metrics, as the space of learned poses
    # is different from the space of colmap poses.
    if args.type_to_eval == 'train':
        eval_pose_param_net = learned_pose_param_net
    else:
        with torch.no_grad():
            # compuate a scale between two learned traj and colmap traj
            init_c2ws = scene_eval.c2ws
            learned_c2ws_train = torch.stack([learned_pose_param_net(i) for i in range(scene_train.N_imgs)])  # (N, 4, 4)
            colmap_c2ws_train = scene_train.c2ws  # (N, 4, 4)
            init_c2ws, scale_colmap2est = align_scale_c2b_use_a2b(colmap_c2ws_train, learned_c2ws_train, init_c2ws)

        eval_pose_param_net = LearnPose(scene_eval.N_imgs, args.opt_eval_R, args.opt_eval_t, init_c2ws)
        if args.multi_gpu:
            eval_pose_param_net = torch.nn.DataParallel(eval_pose_param_net).to(device=my_devices)
        else:
            eval_pose_param_net = eval_pose_param_net.to(device=my_devices)

    '''Set Optimiser'''
    optimizer_eval_pose = torch.optim.Adam(eval_pose_param_net.parameters(), lr=args.opt_eval_lr)
    scheduler_eval_pose = torch.optim.lr_scheduler.MultiStepLR(optimizer_eval_pose,
                                                               milestones=args.eval_pose_milestones,
                                                               gamma=args.eval_pose_lr_gamma)

    '''Optimise eval poses'''
    if args.type_to_eval != 'train':
        for epoch_i in tqdm(range(args.opt_pose_epoch), desc='optimising eval'):
            mean_losses = opt_eval_pose_one_epoch(model, focal_net, eval_pose_param_net, scene_eval, optimizer_eval_pose,
                                                  my_devices)
            opt_L2_loss = mean_losses['L2']
            opt_pose_psnr = mse2psnr(opt_L2_loss)
            scheduler_eval_pose.step()

            writer.add_scalar('opt/mse', opt_L2_loss, epoch_i)
            writer.add_scalar('opt/psnr', opt_pose_psnr, epoch_i)

            logger.info('{0:6d} ep: Opt: L2 loss: {1:.4f}, PSNR: {2:.3f}'.format(epoch_i, opt_L2_loss, opt_pose_psnr))
            tqdm.write('{0:6d} ep: Opt: L2 loss: {1:.4f}, PSNR: {2:.3f}'.format(epoch_i, opt_L2_loss, opt_pose_psnr))

    with torch.no_grad():
        '''Compute ATE'''
        stats_tran, stats_rot, stats_scale = eval_one_epoch_traj(scene_train, learned_pose_param_net)
        print('------------------ ATE statistic ------------------')
        print('Traj Err: translation: {0:.6f}, rotation: {1:.2f} deg, scale: {2:.2f}'.format(stats_tran['mean'],
                                                                                             stats_rot['mean'],
                                                                                             stats_scale['mean']))
        print('-------------------------------------------------')

        '''Final Render'''
        result = eval_one_epoch_img(scene_eval, model, focal_net, eval_pose_param_net, my_devices, args, logger)
        imgs = result['imgs']
        depths = result['depths']

        '''Write to folder'''
        imgs = (imgs.cpu().numpy() * 255).astype(np.uint8)
        depths = (depths.cpu().numpy() * 10).astype(np.uint8)

        for i in range(scene_eval.N_imgs):
            imageio.imwrite(os.path.join(img_out_dir, str(i).zfill(4) + '.png'), imgs[i])
            imageio.imwrite(os.path.join(depth_out_dir, str(i).zfill(4) + '.png'), depths[i])

        imageio.mimwrite(os.path.join(video_out_dir, 'img.mp4'), imgs, fps=30, quality=9)
        imageio.mimwrite(os.path.join(video_out_dir, 'depth.mp4'), depths, fps=30, quality=9)

        imageio.mimwrite(os.path.join(video_out_dir, 'img.gif'), imgs, fps=30)
        imageio.mimwrite(os.path.join(video_out_dir, 'depth.gif'), depths, fps=30)
        return


if __name__ == '__main__':
    args = parse_args()
    set_randomness(args)
    main(args)
