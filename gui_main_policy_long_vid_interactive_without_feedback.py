from __future__ import print_function
import h5py
import sys
import argparse
import numpy as np
import time
import datetime
import copy
from tabulate import tabulate
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli
import matplotlib.pyplot as plt
from utils import Logger, read_json, write_json, save_checkpoint
from models import *
from rewards_interactive_without_feedback import compute_reward
import vsum_tools
import timeit
from tqdm import tqdm
import csv
import random


def epoch_plot(epoch , y , save_file):
    x=[]
    for i in range(epoch):
        x.append(i+1)
    plt.plot(x, y)
    plt.title("Epoch Vs Reward")
    plt.ylabel('Reward')
    plt.xlabel('Epochs')
    #plt.show()
    plt.savefig('./log/plot_epochs_vs_reward_' + save_file)
    plt.close()

def update_summary_indexes(idxes, q_val, summary_length):
    q_val = torch.squeeze(q_val)
    sorted_q_val, indices = q_val.sort(descending=True)
    top_idices, _ = indices[:summary_length].sort()
    #print (q_val)
    top_idices = list(top_idices.cpu().detach().numpy())
    #print (top_idices)
    #print (idxes)
    return [val for idd,val in enumerate(idxes) if idd in top_idices]

def get_past_summary_idx(win_start, idxes):
    return [val for idx,val in enumerate(idxes) if val < win_start]

def get_future_summary_idx(win_end, idxes):
    return [val for idx,val in enumerate(idxes) if val > win_end]

def get_initial_summary_uniform(video_length, summary_length):
    print ('-----> Uniform initialization: ')
    idxes = range(0, video_length, video_length//summary_length)

    if len(idxes) >= summary_length:
        idxes = idxes[0:summary_length]
    return idxes

def get_initial_summary_random(video_length, summary_length, negative_feedback_idxes):
    print ('-----> Random initialization: ')
    numlst = []
    while len(numlst) < summary_length:
        rnd = random.randint(0, video_length)
        if rnd in numlst or rnd in negative_feedback_idxes:
            continue
        else:
    	    numlst += [rnd]
    numlst.sort()
    # write initial summary in one one_hot
    file_name = 'initial_random_summary.txt'
    print (numlst)
    write_one_hot_summary(file_name, numlst, video_length)

    return numlst

def get_existing_summary(file_name):
    print ('-----> Initialization of summary from previous generated summary ')
    with open(file_name) as f:
        labels=f.read().split('\n')
    labels=labels[:-1]
    idxes = [ii for ii, ee in enumerate(labels) if ee == '1']
    return idxes

def write_one_hot_summary(file_name, idxes, video_length):
    summary_file = open(file_name,"w")
    summary_one_hot = np.zeros([video_length], dtype='int64')
    summary_one_hot[idxes] = 1
    summary_one_hot = list(summary_one_hot)
    for frame_select in summary_one_hot:
        summary_file.write(str(frame_select))
        summary_file.write("\n")

input_dim = 512
hidden_dim = 256
num_layers = 1
rnn_cell = 'lstm'
# Optimization options
lr = 1e-05
weight_decay = 1e-05
max_epoch = 1
stepsize = 30
gamma_ = 0.1
num_episode = 5
beta = 0.01
# Misc
seed = 1
gpu = '0'
save_dir = 'log'

GAMMA = .99

torch.manual_seed(seed)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
use_gpu = torch.cuda.is_available()
#if args.use_cpu: use_gpu = False
np.random.seed(seed=5)

def generate_normal_summary(dataset_name, video_name):
    #Hyperparas

    if not os.path.exists('log'):
        os.makedirs('log')
        
    print ('Hidden Units in LSTM:   ' + str(hidden_dim))
    summary_length = 600
    subshot_length = 200
    pseudo_batch_size = 2
    max_passes = 4
    summary_update_count = 0

    # Variables to store best summary
    my_epis_reward = 0
    my_old_reward = 0
    best_f_score = 0

    if use_gpu:
        print("Currently using GPU {}".format(gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
    else:
        print("Currently using CPU")

    print("Initialize dataset {}".format(dataset_name))

    dataset = h5py.File('datasets/'+dataset_name+'_features.h5', 'r')
    num_videos = len(dataset.keys())
    #splits = read_json(args.split)
    #assert args.split_id < len(splits), "split_id (got {}) exceeds {}".format(args.split_id, len(splits))
    # split = splits[args.split_id]
    # train_keys = split['train_keys']
    # test_keys = split['test_keys']
    if dataset_name == 'UTE':
        train_keys = [video_name]
    else:
        train_keys = [video_name+'.mp4']
    #print("# total videos {}. # train videos {}. # test videos {}".format(num_videos, len(train_keys), len(test_keys)))

    print("Initialize model")
    model = DSN(in_dim=input_dim, hid_dim=hidden_dim, num_layers = num_layers, cell=rnn_cell)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)

    if stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gamma_)

    # if args.resume:
    #     print("Loading checkpoint from '{}'".format(args.resume))
    #     checkpoint = torch.load(args.resume)
    #     model.load_state_dict(checkpoint)
    #
    # else:
    start_epoch = 0

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    print("==> Start training")
    start_time = time.time()

    baselines = {key: 0. for key in train_keys} # baseline rewards for videos

    reward_writers = {key: [] for key in train_keys} # record reward changes for each video

    final_epoch_rewards = []

    for epoch in range(start_epoch, max_epoch):
        model.train()
        vid_idxes = np.arange(len(train_keys))
        epoch_reward = torch.zeros([], dtype=torch.float32).cuda()
        np.random.shuffle(vid_idxes)
        if epoch%3==0:
            model = copy.deepcopy(model)

        for vid_idx in vid_idxes:
            vid_name = train_keys[vid_idx]
            print ('Video Name: ',vid_name)
            full_video = dataset[vid_name]['features_c3d'][...]
            print ('--------------Features Normalized----------------')
            full_video = full_video/np.max(full_video)
            video_length = full_video.shape[0]
            print ('Total Frames: ', video_length)
            #idxes = get_initial_summary_random(video_length, summary_length, idx_to_ignore_for_init)
            idxes = get_initial_summary_uniform(video_length, summary_length)
            vid_reward = torch.zeros([],dtype=torch.float32).cuda()

            for _ in range(max_passes):
                window_position = 0
                # pseudo batch
                vid_pass_reward = torch.zeros([], dtype=torch.float32).cuda()
                pseudo_batch_counter = 0
                pseudo_batch_cost = torch.zeros([], dtype=torch.float32).cuda()
                slide_count = 0
                while window_position <= video_length-subshot_length:
                    slide_count += 1
                    current_window_idx = range(window_position, window_position + subshot_length)
                    idx_past = get_past_summary_idx(current_window_idx[0], idxes)
                    idx_fut = get_future_summary_idx(current_window_idx[-1], idxes)
                    idxes_without_sliding_window = idxes
                    idxes = idx_past+ list(current_window_idx) + idx_fut

                    seq = full_video[idxes,:]
                    seq = torch.from_numpy(seq).unsqueeze(0)
                    if use_gpu: seq = seq.cuda()
                    probs = model(seq) # output shape (1, seq_len, 1)

                    cost = beta * (probs.mean()-0.5)**2
                    m = Bernoulli(probs)
                    epis_rewards = []
                    for _ in range(num_episode):
                        actions = m.sample()
                        log_probs = m.log_prob(actions)
                        reward = compute_reward(full_video, seq, actions, use_gpu=use_gpu)
                        expected_reward = log_probs.mean() * (reward - baselines[vid_name])
                        cost -= expected_reward # minimize negative expected reward
                        epis_rewards.append(reward.item())

                    pseudo_batch_cost += cost
                    pseudo_batch_counter += 1
                    vid_pass_reward += np.mean(epis_rewards)
                    my_epis_reward = np.mean(epis_rewards)

                    if my_epis_reward > my_old_reward:
                        idxes = update_summary_indexes(idxes, probs, summary_length)
                        my_old_reward = my_epis_reward
                        summary_update_count += 1
                        print ('SUMMARY UPDATE NUMBER:  ----------------->>', summary_update_count)
                        print ('Current Reward: ', my_epis_reward)
                        #summary_file = open(vid_name + "_policy_grad_summary_length_" +str(summary_length)+"_subshot_size_" + str(subshot_length) + "_hidden_dim_" + str(hidden_dim) + "_summary_alin_sitting_in_train_excluded.txt","w")
                        #summary_file_name = vid_name + "_policy_grad_summary_length_" +str(summary_length)+"_subshot_size_" + str(subshot_length) + "_hidden_dim_" + str(hidden_dim) + "_summary_dark_events_included_with_previous_generated_summary.txt"
                        summary_file_name = 'output_summary_without_feedback/'+ vid_name+ '_'+dataset_name + '_policy_grad_summary_length_' +str(summary_length)+'_subshot_size_' + str(subshot_length) + '_hidden_dim_' + str(hidden_dim) + '_summary_without_feedback.txt'
                        write_one_hot_summary(summary_file_name, idxes, video_length)

                    else:
                        idxes = idxes_without_sliding_window

                    #idxes = get_initial_summary(video_length, summary_length)

                    if pseudo_batch_counter % pseudo_batch_size ==0:
                        pseudo_batch_cost = pseudo_batch_cost/pseudo_batch_size
                        # weight update actor
                        # clear gradient wrt parameters
                        optimizer.zero_grad()
                        # getting gradient wrt parameters
                        pseudo_batch_cost.backward()
                        # gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        # updating parameres
                        optimizer.step()

                        reward_writers[vid_name].append(int(np.mean(epis_rewards)))
                        pseudo_batch_counter = 0
                        pseudo_batch_cost = torch.zeros([], dtype = torch.float32).cuda()

                    window_position = window_position + subshot_length
                    print ('Window Position: ', window_position)

                vid_pass_reward = vid_pass_reward/slide_count
                vid_reward += vid_pass_reward
                baselines[vid_name] = 0.9 * baselines[vid_name] + 0.1 * vid_pass_reward

            vid_reward = vid_reward/max_passes
            epoch_reward += vid_reward
            #print ("Final reward of the video is: ", vid_reward.data.tolist())
        epoch_reward = epoch_reward/(len(vid_idxes))
        final_epoch_rewards.append(epoch_reward)
        print("epoch {}/{}\t reward {}\t".format(epoch+1, max_epoch, epoch_reward))


        epoch_plot(epoch+1, final_epoch_rewards , 'Plot_without_feedback_'+dataset_name+'_'+ video_name +'_model_epoch_' + str(epoch+1) + '_passes_' + str(max_passes) + '_subshot_len_' + str(subshot_length) + '_batch_size_' + str(pseudo_batch_size) + '.png')

        model_state_dict = model.module.state_dict() if use_gpu else model.state_dict()
        model_save_path = osp.join(save_dir,dataset_name+'_'+ video_name + '_policy_grad_without_feedback_epoch_' + str(epoch+1) + '_passes_' + str(max_passes) + '_subshot_len_' + str(subshot_length) + '_batch_size_' + str(pseudo_batch_size) +'.pth.tar')
        save_checkpoint(model_state_dict, model_save_path)
        print("Model saved to {}".format(model_save_path))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
    print ("################################")
    dataset.close()
    print('Find the one hot summary here: {}'.format(summary_file_name))
    return summary_file_name
