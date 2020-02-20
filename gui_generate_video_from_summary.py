import cv2
import os
import argparse

def generate_video_from_summary(summary_file_path, dataset, video_name):
    if dataset == 'Disney':
        input_vid_dir = "/media/enigma/f0762f3b-20d1-42a7-9fe1-60385c4a8a3e/video_summarization/actor_critic/dataset/Disney/full_videos/"
    if dataset == 'UTE':
        input_vid_dir = "/media/enigma/f0762f3b-20d1-42a7-9fe1-60385c4a8a3e/video_summarization/actor_critic/dataset/UTE/full_videos/"
    if dataset == 'HUJI':
        input_vid_dir = "/media/enigma/f0762f3b-20d1-42a7-9fe1-60385c4a8a3e/video_summarization/actor_critic/dataset/HUJI/full_videos/"

    sampling_rate = 15 # for C3D
    frames = []
    summary_file = summary_file_path
    input_video = input_vid_dir + video_name + '.mp4'
    output_video =  summary_file_path[:-4] + ".mp4"
    fps = 15.0
    a = 1
    #module to add when using subsampling
    with open(summary_file) as f:
        line_num = 0
        for line in f:
            line_num +=1
            if int(line) == 1:
                for i in range(sampling_rate):
                    frames.append(line_num * sampling_rate - i)

    print (len(frames))
    print ('Input video is: {}'.format(input_video))
    print ('Output video is: {}'.format(output_video))
#     print (frames)
#     fourcc = cv2.CV_FOURCC('X','V','I','D')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if  a !=0:
        vidcap = cv2.VideoCapture(input_video)
        success,image = vidcap.read()
        print (success)
        height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width  = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        print ("height:", height)
        print ("width:", width)
        out = cv2.VideoWriter(output_video, fourcc, fps, (int(width),int(height)))

        count = 1
        act = 0
        while success:
            success,image = vidcap.read()
            if count in frames:
                act +=1
                out.write(image)
#                 print '.',
            count += 1
            if count % 10000 ==0:
                print (count)

    print (count)
    print (len(frames))
    out.release()
    print ("Find the video: ",output_video)
