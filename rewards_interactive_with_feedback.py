import torch
import sys
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

def compute_reward(full_video, seq, actions, positive_feedback_idxes, negative_feedback_idxes, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False):
    """
    Compute diversity reward and representativeness reward

    Args:
        seq: sequence of features, shape (1, seq_len, dim)
        actions: binary action sequence, shape (1, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU
    """

    # seting up the Variables
    _seq = seq.detach().squeeze()
    normed_seq = _seq/ _seq.norm(p=2, dim=1, keepdim=True)
    full_video = torch.from_numpy(full_video)
    full_video = full_video.cuda()
    normed_full_video = full_video/ full_video.norm(p=2, dim=1, keepdim=True)
    _actions = actions.detach()
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1

    if num_picks == 0:
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward
    if num_picks == 1:
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward
    else:
        # compute diversity reward
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())
        dissim_submat = dissim_mat[pick_idxs,:][:, pick_idxs]
        if ignore_far_sim:
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))

        # Computation for negative and positive read_feedback
        positive_feedback_idxes = torch.tensor(positive_feedback_idxes)
        negative_feedback_idxes = torch.tensor(negative_feedback_idxes)
        positive_feedback_feat = full_video[positive_feedback_idxes,:]
        negative_feedback_feat = full_video[negative_feedback_idxes,:]
        # Similarity with positive feedback
        positive_feedback_feat = positive_feedback_feat/ positive_feedback_feat.norm(p=2, dim=1, keepdim=True)
        sim_mat = torch.matmul(normed_seq, positive_feedback_feat.t())
        sim_submat = sim_mat[pick_idxs,:][:,:]
        reward_sim_with_positive = sim_submat.sum()/ (num_picks * (len(positive_feedback_idxes)-1.) )
        # Dissimiilarity with negative feedback
        negative_feedback_feat = negative_feedback_feat/ negative_feedback_feat.norm(p=2, dim=1, keepdim=True)
        dissim_mat = 1. - torch.matmul(normed_seq, negative_feedback_feat.t())
        dissim_submat = dissim_mat[pick_idxs,:][:,:]
        reward_dissim_with_negative = dissim_submat.sum()/ (num_picks * (len(negative_feedback_idxes)-1.) )

        # compute indicative reward
        summary_feat = normed_seq[pick_idxs, :]
        n_summary_frames, feat_dimen = summary_feat.size()
        full_video = full_video[[i for i in range(0, full_video.size(0) ,2)] ,:]
        n_total_frames = full_video.size(0)
        # code to calculate clossest summary to each input frames in batches of dataset
        batch_ka_size = 2000
        final_distances = torch.rand(1).cuda()
        for II in range(0, n_total_frames, batch_ka_size):
            batch_vid_seq = full_video[II:II+batch_ka_size,:]
            batch_distances = compute_indicativeness_in_batch(summary_feat, batch_vid_seq)
            final_distances = torch.cat((final_distances, batch_distances),0)

        final_distances = final_distances[1:]
        reward_ind = -final_distances.mean()
        print (4*reward_dissim_with_negative, 3*reward_sim_with_positive)
        #reward = 4*reward_dissim_with_negative + .5* reward_div + .2*reward_ind # interactive reward
        reward = 4*reward_dissim_with_negative + 3*reward_sim_with_positive + .5* reward_div + .2*reward_ind # interactive reward
        #print (1.2*reward_div, .3*reward_ind)
        #reward =  1.2*reward_div + .3*reward_ind # normal summary

    return reward
