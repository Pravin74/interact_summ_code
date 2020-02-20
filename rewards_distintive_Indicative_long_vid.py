import torch
from numpy.linalg import norm
import sys
import numpy as np
import scipy.spatial as sp
def compute_indicativeness_in_batch(summary_feat, full_vid_seq):
    n_summary_frames , feat_dimen = summary_feat.size()
    n_total_frames_batch = full_vid_seq.size(0)
    full_vid_seq_repeat = full_vid_seq.repeat(n_summary_frames,1,1)
    del full_vid_seq
    repeat_summary_features = torch.zeros(n_summary_frames, n_total_frames_batch , feat_dimen).cuda()
    for jj in range(n_summary_frames):
        repeat_summary_features[jj,:,:] = summary_feat[jj,:].repeat(n_total_frames_batch,1)
    repeat_summary_features = repeat_summary_features - full_vid_seq_repeat
    #del repeat_summary_features
    del full_vid_seq_repeat
    difference_matrix_l2_norm = torch.norm(repeat_summary_features,2,dim=2)
    del repeat_summary_features
    closest_frames, _ = torch.max(difference_matrix_l2_norm, dim=0)
    return closest_frames

def compute_reward(seq, actions, full_vid_seq, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False):
    """
    Compute diversity reward and representativeness reward
    Args:
        seq: sequence of features, shape (1, seq_len, dim)
        actions: binary action sequence, shape (1, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU
    """
    _seq = seq.detach()
    _actions = actions.detach()
    #print (_actions.shape)
    #print (_seq.shape)
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1

    if num_picks == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward

    _seq = _seq.squeeze()
    n = _seq.size(0)
    # compute diversity reward
    if num_picks == 1:
        reward_div = torch.tensor(0.)
        if use_gpu: reward_div = reward_div.cuda()
    else:
        # Calculations for distinctive reward
        #repeating the [x1-x1, x1-x2, x1-x3][x2-x1, x2-x2, x2-x3][x3-x1, x3-x2, x3-x3]
        repeat_seq = _seq.repeat(n,1,1)
        repeat_feat = torch.zeros(repeat_seq.size()).cuda()
        for ii in range(n):
            repeat_feat[ii,:,:] = _seq[ii,:].repeat(n,1)
        vector_subtract =  repeat_feat - repeat_seq
        vector_subtract_l2_norm = torch.norm(vector_subtract,2,dim=2)
        distinctive_submat = vector_subtract_l2_norm[pick_idxs,:][:,pick_idxs]
        if ignore_far_sim:
            # ignore temporally distant similarity
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            distinctive_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_distinctive = distinctive_submat.sum() / (num_picks * (num_picks + 1 )) # diversity reward [Eq.3]

    # compute indicative reward
    # vid feat are N*F to S*N*F and summary feature are S*F to S*N*F (repeat each summary feature to N(total video frames) times )
    summary_feat = _seq[pick_idxs, :]
    n_summary_frames , feat_dimen = summary_feat.size()
    full_vid_seq = torch.from_numpy(full_vid_seq)
    full_vid_seq = full_vid_seq[[i for i in range(0, full_vid_seq.size(0) ,2)] ,:]
    full_vid_seq = full_vid_seq.cuda()
    n_total_frames = full_vid_seq.size(0)
    # code to calculate clossest summary to each input frames in batches of dataset
    batch_ka_size = 1000
    final_distances = torch.rand(1).cuda()
    for II in range(0,n_total_frames, batch_ka_size):
        batch_vid_seq = full_vid_seq[II:II+batch_ka_size,:]
        batch_distances = compute_indicativeness_in_batch(summary_feat, batch_vid_seq)
        final_distances = torch.cat((final_distances, batch_distances),0)

    final_distances = final_distances[1:]
    reward_ind = -final_distances.mean()
    #print (.6*reward_distinctive, .4*reward_ind)
    return (.6*reward_distinctive + .4*reward_ind)
