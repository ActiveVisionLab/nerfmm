import sys
import os
import argparse
from pathlib import Path

sys.path.append(os.path.join(sys.path[0], '../..'))

import open3d as o3d
from utils.vis_cam_traj import draw_camera_frustum_geometry

import torch
import numpy as np

from dataloader.with_colmap import DataLoaderWithCOLMAP
from utils.training_utils import set_randomness, load_ckpt_to_net
from utils.align_traj import align_ate_c2b_use_a2b, pts_dist_max
from utils.comp_ate import compute_ate
from models.intrinsics import LearnFocal
from models.poses import LearnPose


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='./data_dir/nerfmm_release_data',
                        help='folder contains various scenes')
    parser.add_argument('--scene_name', type=str, default='LLFF/fern')

    parser.add_argument('--learn_focal', default=False, type=bool)
    parser.add_argument('--fx_only', default=False, type=bool)
    parser.add_argument('--focal_order', default=2, type=int)

    parser.add_argument('--learn_R', default=False, type=bool)
    parser.add_argument('--learn_t', default=False, type=bool)

    parser.add_argument('--init_pose_colmap', default=False, type=bool,
                        help='set this to True if the nerfmm model is trained from COLMAP init.')
    parser.add_argument('--init_focal_colmap', default=False, type=bool,
                        help='set this to True if the nerfmm model is trained from COLMAP init.')

    parser.add_argument('--resize_ratio', type=int, default=8, help='lower the image resolution with this ratio')

    parser.add_argument('--ATE_align', type=bool, default=True)
    parser.add_argument('--train_img_num', type=int, default=-1, help='num of images to train')
    parser.add_argument('--train_skip', type=int, default=1, help='skip every this number of imgs')

    parser.add_argument('--rand_seed', type=int, default=17)
    parser.add_argument('--true_rand', type=bool, default=False)

    parser.add_argument('--ckpt_dir', type=str, default='')
    return parser.parse_args()


def main(args):
    my_devices = torch.device('cpu')

    '''Create Folders'''
    pose_out_dir = Path(os.path.join(args.ckpt_dir, 'pose_out'))
    pose_out_dir.mkdir(parents=True, exist_ok=True)

    '''Get COLMAP poses'''
    scene_train = DataLoaderWithCOLMAP(base_dir=args.base_dir,
                                       scene_name=args.scene_name,
                                       data_type='train',
                                       res_ratio=args.resize_ratio,
                                       num_img_to_load=args.train_img_num,
                                       skip=args.train_skip,
                                       use_ndc=True,
                                       load_img=False)

    # scale colmap poses to unit sphere
    ts_colmap = scene_train.c2ws[:, :3, 3]  # (N, 3)
    scene_train.c2ws[:, :3, 3] /= pts_dist_max(ts_colmap)
    scene_train.c2ws[:, :3, 3] *= 2.0

    '''Load scene meta'''
    H, W = scene_train.H, scene_train.W
    colmap_focal = scene_train.focal

    print('Intrinsic: H: {0:4d}, W: {1:4d}, COLMAP focal {2:.2f}.'.format(H, W, colmap_focal))

    '''Model Loading'''
    if args.init_focal_colmap:
        focal_net = LearnFocal(H, W, args.learn_focal, args.fx_only, order=args.focal_order, init_focal=colmap_focal)
    else:
        focal_net = LearnFocal(H, W, args.learn_focal, args.fx_only, order=args.focal_order)
        # only load learned focal if we do not init with colmap focal
        focal_net = load_ckpt_to_net(os.path.join(args.ckpt_dir, 'latest_focal.pth'), focal_net, map_location=my_devices)
    fxfy = focal_net(0)
    print('COLMAP focal: {0:.2f}, learned fx: {1:.2f}, fy: {2:.2f}'.format(colmap_focal, fxfy[0].item(), fxfy[1].item()))

    if args.init_pose_colmap:
        pose_param_net = LearnPose(scene_train.N_imgs, args.learn_R, args.learn_t, scene_train.c2ws)
    else:
        pose_param_net = LearnPose(scene_train.N_imgs, args.learn_R, args.learn_t, None)
    pose_param_net = load_ckpt_to_net(os.path.join(args.ckpt_dir, 'latest_pose.pth'), pose_param_net, map_location=my_devices)

    '''Get all poses in (N, 4, 4)'''
    c2ws_est = torch.stack([pose_param_net(i) for i in range(scene_train.N_imgs)])  # (N, 4, 4)
    c2ws_cmp = scene_train.c2ws  # (N, 4, 4)

    # scale estimated poses to unit sphere
    ts_est = c2ws_est[:, :3, 3]  # (N, 3)
    c2ws_est[:, :3, 3] /= pts_dist_max(ts_est)
    c2ws_est[:, :3, 3] *= 2.0

    '''Define camera frustums'''
    frustum_length = 0.1
    est_traj_color = np.array([39, 125, 161], dtype=np.float32) / 255
    cmp_traj_color = np.array([249, 65, 68], dtype=np.float32) / 255

    '''Align est traj to colmap traj'''
    c2ws_est_to_draw_align2cmp = c2ws_est.clone()
    if args.ATE_align:  # Align learned poses to colmap poses
        c2ws_est_aligned = align_ate_c2b_use_a2b(c2ws_est, c2ws_cmp)  # (N, 4, 4)
        c2ws_est_to_draw_align2cmp = c2ws_est_aligned

        # compute ate
        stats_tran_est, stats_rot_est, _ = compute_ate(c2ws_est_aligned, c2ws_cmp, align_a2b=None)
        print('From est to colmap: tran err {0:.3f}, rot err {1:.2f}'.format(stats_tran_est['mean'],
                                                                             stats_rot_est['mean']))

    frustum_est_list = draw_camera_frustum_geometry(c2ws_est_to_draw_align2cmp.cpu().numpy(), H, W,
                                                    fxfy[0], fxfy[1],
                                                    frustum_length, est_traj_color)
    frustum_colmap_list = draw_camera_frustum_geometry(c2ws_cmp.cpu().numpy(), H, W,
                                                       colmap_focal, colmap_focal,
                                                       frustum_length, cmp_traj_color)

    geometry_to_draw = []
    geometry_to_draw.append(frustum_est_list)
    geometry_to_draw.append(frustum_colmap_list)

    '''o3d for line drawing'''
    t_est_list = c2ws_est_to_draw_align2cmp[:, :3, 3]
    t_cmp_list = c2ws_cmp[:, :3, 3]

    '''line set to note pose correspondence between two trajs'''
    line_points = torch.cat([t_est_list, t_cmp_list], dim=0).cpu().numpy()  # (2N, 3)
    line_ends = [[i, i+scene_train.N_imgs] for i in range(scene_train.N_imgs)]  # (N, 2) connect two end points.
    # line_color = np.zeros((scene_train.N_imgs, 3), dtype=np.float32)
    # line_color[:, 0] = 1.0

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(line_ends)
    # line_set.colors = o3d.utility.Vector3dVector(line_color)

    geometry_to_draw.append(line_set)
    o3d.visualization.draw_geometries(geometry_to_draw)


if __name__ == '__main__':
    args = parse_args()
    set_randomness(args)
    with torch.no_grad():
        main(args)
