from third_party.ATE.compute_trajectory_errors import compute_absolute_error
from third_party.ATE.results_writer import compute_statistics
from utils.lie_group_helper import SO3_to_quat
from utils.align_traj import align_ate_c2b_use_a2b


def compute_ate(c2ws_a, c2ws_b, align_a2b=None):
    """Compuate ate between a and b.
    :param c2ws_a: (N, 3/4, 4) torch
    :param c2ws_b: (N, 3/4, 4) torch
    :param align_a2b: None or 'sim3'. Set to None if a and b are pre-aligned.
    """
    if align_a2b == 'sim3':
        c2ws_a_aligned = align_ate_c2b_use_a2b(c2ws_a, c2ws_b)
        R_a_aligned = c2ws_a_aligned[:, :3, :3].cpu().numpy()
        t_a_aligned = c2ws_a_aligned[:, :3, 3].cpu().numpy()
    else:
        R_a_aligned = c2ws_a[:, :3, :3].cpu().numpy()
        t_a_aligned = c2ws_a[:, :3, 3].cpu().numpy()
    R_b = c2ws_b[:, :3, :3].cpu().numpy()
    t_b = c2ws_b[:, :3, 3].cpu().numpy()

    quat_a_aligned = SO3_to_quat(R_a_aligned)
    quat_b = SO3_to_quat(R_b)

    e_trans, e_trans_vec, e_rot, e_ypr, e_scale_perc = compute_absolute_error(t_a_aligned,quat_a_aligned,
                                                                              t_b, quat_b)
    stats_tran = compute_statistics(e_trans)
    stats_rot = compute_statistics(e_rot)
    stats_scale = compute_statistics(e_scale_perc)

    return stats_tran, stats_rot, stats_scale  # dicts
