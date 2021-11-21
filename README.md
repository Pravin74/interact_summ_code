# Generating Personalized Video Summaries of Day Long Egocentric Videos

The proposed framework facilitates to incorporate interactive user choice in terms of including or excluding the
particular type of content. Our approach can also be adapted to generate summaries of various lengths making it possible to view even 1-minute summaries of one’s entire day. When using the facial saliency-based reward, we show that our approach generates summaries focusing on social interactions, similar to the current state-of-the-art (SOTA).

## Requirements
The code has been tested on:

- Nvidia P5000 GPU
- Ubuntu 16.04 LTS & UBUNTU 18.04 LTS
- Pytorch v1.4.0
- CUDA 9.0 & CUDA 10.1

## Get started
Extract C3D features of Disney, UTE and HUJI dataset or you can download from https://pravin74.github.io/Int-sum/index.html

## How to generate summaries

Download the h5 file of all the datasets in the "datasets" folder from the above link. Rename 'Disney_features_CNN_C3D.h5' to 'Disney_features.h5', and 'UTE_features_C3D.h5' to 'UTE_features.h5' then run GUI.py. Select the appropriate dataset and corresponding video and click on 'Generate summary without feedback'. You will get the normal summary(video summary and a text file of selected frames) in 'output_summary_with_feedback' folder. Then you have to look at the generated summary (without feedback) and select the events you want to include or exclude. Just put the time intervals of the selected positive and negative events in the GUI in MM:SS MM:SS format.
After clicking the on 'Generate Summary with feedback' you will get a customized summary in 'output_summary_with_feedback' folder.  In 'plot_comparison' folder you will get the plots of summary with feedback and without feedback for comparison.
## Citation
```
@article{nagar2021generating,
  title={Generating Personalized Summaries of Day Long Egocentric Videos},
  author={Nagar, Pravin and Rathore, Anuj and Jawahar, CV and Arora, Chetan},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
  publisher={IEEE}
}

@inproceedings{rathore2019generating,
    title={Generating 1 Minute Summaries of Day Long Egocentric Videos},
    author={Rathore, Anuj and Nagar, Pravin and Arora, Chetan and Jawahar, CV},
    booktitle={ACMMM},
    year={2019}
```
