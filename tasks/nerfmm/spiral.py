import sys
import os
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
import imageio

sys.path.append(os.path.join(sys.path[0], '../..'))

from dataloader.with_colmap import DataLoaderWithCOLMAP
from utils.training_utils import set_randomness, load_ckpt_to_net
from utils.pose_utils import create_spiral_poses
from utils.comp_ray_dir import comp_ray_dir_cam_fxfy
from utils.lie_group_helper import convert3x4_4x4
from models.nerf_models import OfficialNerf
from tasks.nerfmm.train import model_render_image
from models.intrinsics import LearnFocal
from models.poses import LearnPose


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

    parser.add_argument('--init_focal_colmap', default=False, type=bool)

    parser.add_argument('--resize_ratio', type=int, default=4, help='lower the image resolution with this ratio')
    parser.add_argument('--num_rows_eval_img', type=int, default=10, help='split a high res image to rows in eval')
    parser.add_argument('--hidden_dims', type=int, default=128, help='network hidden unit dimensions')
    parser.add_argument('--num_sample', type=int, default=128, help='number samples along a ray')

    parser.add_argument('--pos_enc_levels', type=int, default=10, help='number of freqs for positional encoding')
    parser.add_argument('--pos_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--use_dir_enc', type=bool, default=True, help='use pos enc for view dir?')
    parser.add_argument('--dir_enc_levels', type=int, default=4, help='number of freqs for positional encoding')
    parser.add_argument('--dir_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--rand_seed', type=int, default=17)
    parser.add_argument('--true_rand', type=bool, default=False)

    parser.add_argument('--spiral_mag_percent', type=float, default=50, help='for np.percentile')
    parser.add_argument('--spiral_axis_scale', type=float, default=[1.0, 1.0, 1.0], nargs=3,
                        help='applied on top of percentile, useful in zoom in motion')
    parser.add_argument('--N_img_per_circle', type=int, default=60)
    parser.add_argument('--N_circle_traj', type=int, default=2)

    parser.add_argument('--train_img_num', type=int, default=-1, help='num of images to train, -1 for all')
    parser.add_argument('--train_skip', type=int, default=1, help='skip every this number of imgs')

    parser.add_argument('--ckpt_dir', type=str, default='')
    return parser.parse_args()


def test_one_epoch(H, W, focal_net, c2ws, near, far, model, my_devices, args):
    model.eval()
    focal_net.eval()

    fxfy = focal_net(0)
    ray_dir_cam = comp_ray_dir_cam_fxfy(H, W, fxfy[0], fxfy[1])
    t_vals = torch.linspace(near, far, args.num_sample, device=my_devices)  # (N_sample,) sample position
    N_img = c2ws.shape[0]

    rendered_img_list = []
    rendered_depth_list = []

    for i in tqdm(range(N_img)):
        c2w = c2ws[i].to(my_devices)  # (4, 4)

        # split an image to rows when the input image resolution is high
        rays_dir_cam_split_rows = ray_dir_cam.split(args.num_rows_eval_img, dim=0)
        rendered_img = []
        rendered_depth = []
        for rays_dir_rows in rays_dir_cam_split_rows:
            render_result = model_render_image(c2w, rays_dir_rows, t_vals, near, far, H, W, fxfy,
                                               model, False, 0.0, args, rgb_act_fn=torch.sigmoid)
            rgb_rendered_rows = render_result['rgb']  # (num_rows_eval_img, W, 3)
            depth_map = render_result['depth_map']  # (num_rows_eval_img, W)

            rendered_img.append(rgb_rendered_rows)
            rendered_depth.append(depth_map)

        # combine rows to an image
        rendered_img = torch.cat(rendered_img, dim=0)  # (H, W, 3)
        rendered_depth = torch.cat(rendered_depth, dim=0)  # (H, W)

        # for vis
        rendered_img_list.append(rendered_img)
        rendered_depth_list.append(rendered_depth)

    rendered_img_list = torch.stack(rendered_img_list)  # (N, H, W, 3)
    rendered_depth_list = torch.stack(rendered_depth_list)  # (N, H, W)

    result = {
        'imgs': rendered_img_list,
        'depths': rendered_depth_list,
    }
    return result


def main(args):
    my_devices = torch.device('cuda:' + str(args.gpu_id))

    '''Create Folders'''
    test_dir = Path(os.path.join(args.ckpt_dir, 'render_spiral'))
    img_out_dir = Path(os.path.join(test_dir, 'img_out'))
    depth_out_dir = Path(os.path.join(test_dir, 'depth_out'))
    video_out_dir = Path(os.path.join(test_dir, 'video_out'))
    test_dir.mkdir(parents=True, exist_ok=True)
    img_out_dir.mkdir(parents=True, exist_ok=True)
    depth_out_dir.mkdir(parents=True, exist_ok=True)
    video_out_dir.mkdir(parents=True, exist_ok=True)

    '''Scene Meta'''
    scene_train = DataLoaderWithCOLMAP(base_dir=args.base_dir,
                                       scene_name=args.scene_name,
                                       data_type='train',
                                       res_ratio=args.resize_ratio,
                                       num_img_to_load=args.train_img_num,
                                       skip=args.train_skip,
                                       use_ndc=args.use_ndc,
                                       load_img=False)

    H, W = scene_train.H, scene_train.W
    colmap_focal = scene_train.focal
    near, far = scene_train.near, scene_train.far

    print('Intrinsic: H: {0:4d}, W: {1:4d}, COLMAP focal {2:.2f}.'.format(H, W, colmap_focal))
    print('near: {0:.1f}, far: {1:.1f}.'.format(near, far))

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

    if args.init_focal_colmap:
        focal_net = LearnFocal(H, W, args.learn_focal, args.fx_only, order=args.focal_order, init_focal=colmap_focal)
    else:
        focal_net = LearnFocal(H, W, args.learn_focal, args.fx_only, order=args.focal_order)
    if args.multi_gpu:
        focal_net = torch.nn.DataParallel(focal_net).to(device=my_devices)
    else:
        focal_net = focal_net.to(device=my_devices)
    # do not load learned focal if we use colmap focal
    if not args.init_focal_colmap:
        focal_net = load_ckpt_to_net(os.path.join(args.ckpt_dir, 'latest_focal.pth'), focal_net, map_location=my_devices)
    fxfy = focal_net(0)
    print('COLMAP focal: {0:.2f}, learned fx: {1:.2f}, fy: {2:.2f}'.format(colmap_focal, fxfy[0].item(), fxfy[1].item()))

    pose_param_net = LearnPose(scene_train.N_imgs, args.learn_R, args.learn_t, None)
    if args.multi_gpu:
        pose_param_net = torch.nn.DataParallel(pose_param_net).to(device=my_devices)
    else:
        pose_param_net = pose_param_net.to(device=my_devices)
    pose_param_net = load_ckpt_to_net(os.path.join(args.ckpt_dir, 'latest_pose.pth'), pose_param_net, map_location=my_devices)

    learned_poses = torch.stack([pose_param_net(i) for i in range(scene_train.N_imgs)])

    '''Generate camera traj'''
    # This spiral camera traj code is modified from https://github.com/kwea123/nerf_pl.
    # hardcoded, this is numerically close to the formula
    # given in the original repo. Mathematically if near=1
    # and far=infinity, then this number will converge to 4
    N_novel_imgs = args.N_img_per_circle * args.N_circle_traj
    focus_depth = 3.5
    radii = np.percentile(np.abs(learned_poses.cpu().numpy()[:, :3, 3]), args.spiral_mag_percent, axis=0)  # (3,)
    radii *= np.array(args.spiral_axis_scale)
    c2ws = create_spiral_poses(radii, focus_depth, n_circle=args.N_circle_traj, n_poses=N_novel_imgs)
    c2ws = torch.from_numpy(c2ws).float()  # (N, 3, 4)
    c2ws = convert3x4_4x4(c2ws)  # (N, 4, 4)

    '''Render'''
    result = test_one_epoch(H, W, focal_net, c2ws, near, far, model, my_devices, args)
    imgs = result['imgs']
    depths = result['depths']

    '''Write to folder'''
    imgs = (imgs.cpu().numpy() * 255).astype(np.uint8)
    depths = (depths.cpu().numpy() * 200).astype(np.uint8)  # far is 1.0 in NDC

    for i in range(c2ws.shape[0]):
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
    with torch.no_grad():
        main(args)
