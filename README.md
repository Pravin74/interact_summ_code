# Generating Personalized Video Summaries of Day Long Egocentric Videos

The proposed framework facilitates to incorporate interactive user choice in terms of including or excluding the
particular type of content. Our approach can also be adapted to generate summaries of various lengths making it possible to view even 1-minute summaries of one’s entire day. When using the facial saliency-based reward, we show that our approach generates summaries focusing on social interactions, similar to the current state-of-the-art (SOTA).
## Get started
Extract C3D features of Disney, UTE and HUJI dataset. or you can downlaod from https://pravin74.github.io/Int-sum/index.html

## How to generate summaries

Create all the folders in the main directory.
Just Run GUI.py. Select the appropriate dataset and corresponding video and click on 'Generate summary without feedback'. You will get the normal summary(video summary and a text file of selected frames) in 'output_summary_with_feedback' folder. Then you have to look at the generated summary (without feedback) and select the events you want to include or exclude. Just put the time intervals of the selected positive and negative events in the GUI in MM:SS MM:SS format. 
After clicking the on 'Generate Summary with feedback' you will get a customized summary in 'output_summary_with_feedback' folder.  In 'plot_comparison' folder you will get the plots of summary with feedback and without feedback for comparison.
## Citation
```
@inproceedings{rathore2019generating,
	title={Generating 1 Minute Summaries of Day Long Egocentric Videos},
	author={Rathore, Anuj and Nagar, Pravin and Arora, Chetan and Jawahar, CV},
	booktitle={ACMMM},
	year={2019}
```
