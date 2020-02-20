# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'video_summ.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import os
from gui_main_policy_long_vid_interactive_without_feedback import generate_normal_summary
from gui_main_policy_long_vid_interactive_with_feedback import generate_summary_with_feedback
from gui_generate_video_from_summary import generate_video_from_summary
from gui_histogram_clustering import plot_normal_and_custom_summary

def secs_to_indexes(temp_in, idxes):
    all_idxes = []
    for row in temp_in:
        print (60*int(row[0:2]) + int(row[3:5]), 60*int(row[6:8])+ int(row[9:11]))
        temp_idx = range(60*int(row[0:2]) + int(row[3:5]), 60*int(row[6:8])+ int(row[9:11]))
        all_idxes = all_idxes + list(temp_idx)
    return all_idxes

def write_feedback_indexes(file_name, positive_feedback_idxes):
    index_file = open(file_name,"w")
    for idx in positive_feedback_idxes:
        index_file.write(str(idx))
        index_file.write("\n")

dataset_global = []
video_name_global = []

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(787, 636)
        self.comboBox_dataset = QtWidgets.QComboBox(Dialog)
        self.comboBox_dataset.setGeometry(QtCore.QRect(190, 90, 151, 41))
        self.comboBox_dataset.setObjectName("comboBox_dataset")
        self.comboBox_dataset.addItem("")
        self.comboBox_dataset.addItem("")
        self.comboBox_dataset.addItem("")
        self.comboBox_dataset.addItem("")
        self.comboBox_video = QtWidgets.QComboBox(Dialog)
        self.comboBox_video.setGeometry(QtCore.QRect(390, 90, 151, 41))
        self.comboBox_video.setObjectName("comboBox_video")
        self.comboBox_video.addItem("")
        self.comboBox_video.addItem("")
        self.comboBox_video.addItem("")
        self.comboBox_video.addItem("")
        self.comboBox_video.addItem("")
        self.comboBox_video.addItem("")
        self.comboBox_video.addItem("")
        self.comboBox_video.addItem("")

        self.pushButton_normal_summary = QtWidgets.QPushButton(Dialog)
        self.pushButton_normal_summary.setGeometry(QtCore.QRect(230, 150, 281, 51))
        self.pushButton_normal_summary.setObjectName("pushButton_normal_summary")
        self.pushButton_normal_summary.clicked.connect(self.btn_click_generate_summary_without_feedback) # Command to connect the relevant button on the button click event

        self.textEdit = QtWidgets.QTextEdit(Dialog)
        self.textEdit.setGeometry(QtCore.QRect(110, 220, 111, 81))
        self.textEdit.setObjectName("textEdit")

        self.textEdit_possitive_feedback_interval = QtWidgets.QTextEdit(Dialog)
        self.textEdit_possitive_feedback_interval.setGeometry(QtCore.QRect(230, 220, 141, 81))
        self.textEdit_possitive_feedback_interval.setObjectName("textEdit_possitive_feedback_interval")
        self.textEdit_possitive_feedback_interval.setPlaceholderText("MM:SS MM:SS")

        self.pushButton_summary_with_feedback = QtWidgets.QPushButton(Dialog)
        self.pushButton_summary_with_feedback.setGeometry(QtCore.QRect(230, 310, 281, 51))
        self.pushButton_summary_with_feedback.setObjectName("pushButton_summary_with_feedback")
        self.pushButton_summary_with_feedback.clicked.connect(self.btn_click_generate_summary_with_feedback)

        self.textEdit_negative_feedback_interval = QtWidgets.QTextEdit(Dialog)
        self.textEdit_negative_feedback_interval.setGeometry(QtCore.QRect(520, 220, 120, 81))
        self.textEdit_negative_feedback_interval.setObjectName("textEdit_negative_feedback_interval")
        self.textEdit_negative_feedback_interval.setPlaceholderText("MM:SS MM:SS")

        self.textEdit_2 = QtWidgets.QTextEdit(Dialog)
        self.textEdit_2.setGeometry(QtCore.QRect(390, 220, 120, 81))
        self.textEdit_2.setObjectName("textEdit_2")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.comboBox_dataset.setItemText(0, _translate("Dialog", "Dataset"))
        self.comboBox_dataset.setItemText(1, _translate("Dialog", "Disney"))
        self.comboBox_dataset.setItemText(2, _translate("Dialog", "UTE"))
        self.comboBox_dataset.setItemText(3, _translate("Dialog", "HUJI"))
        self.comboBox_video.setItemText(0, _translate("Dialog", "Video Name"))
        self.comboBox_video.setItemText(1, _translate("Dialog", "Alin_Day_1"))
        self.comboBox_video.setItemText(2, _translate("Dialog", "P01"))
        self.comboBox_video.setItemText(3, _translate("Dialog", "P02"))
        self.comboBox_video.setItemText(4, _translate("Dialog", "P03"))
        self.comboBox_video.setItemText(5, _translate("Dialog", "P04"))
        self.comboBox_video.setItemText(6, _translate("Dialog", "Alireza_Day_1"))
        self.comboBox_video.setItemText(7, _translate("Dialog", "Michael_Day_2"))
        self.comboBox_video.setItemText(7, _translate("Dialog", "ariel_1"))
        self.pushButton_normal_summary.setText(_translate("Dialog", "Generate Summary without Feedback"))
        self.textEdit.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Positive Feedback Interval</p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"> (in MM:SS)</p></body></html>"))
        self.pushButton_summary_with_feedback.setText(_translate("Dialog", "Generate Summary with Feedback"))
        self.textEdit_negative_feedback_interval.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.textEdit_2.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Negative Feedback in Interval (MM:SS)</p></body></html>"))
        self.textEdit_negative_feedback_interval.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))


    def btn_click_generate_summary_without_feedback(self):
        """ Helper function to generate summary without feedback for the selected dataset and video
        call the python script : main_policy_long_vid_interactive_without_feedback.py with all the necessary flags
        """
        self.dataset = self.comboBox_dataset.currentText() # Read the dataset from the drop box
        self.video_name = self.comboBox_video.currentText() # Read the video_name from the drop box

        # NOTE: Condition to ensure that some video is selected from the dropbox
        # TODO : Throw a warning box in case something or the other is not selected
        if (self.dataset == "" or self.video_name == ""):
            print("Please select dataset and video_name")
            pass
        else:
            #self.normal_summary_path = generate_normal_summary(self.dataset, self.video_name)
            self.normal_summary_path = 'output_summary_without_feedback/Alin_Day_1.mp4_Disney_policy_grad_summary_length_600_subshot_size_200_hidden_dim_256_summary_without_feedback.txt'
            #generate_video_from_summary(self.normal_summary_path, self.dataset, self.video_name)
            print ('Normal Summary Generated:  ', self.normal_summary_path)


    def btn_click_generate_summary_with_feedback(self):
        """This is a function to trigger the button click event for generating summary with a given feedback
        The button click will read the values entered in the text box for positive and negative feedback
        save it, and will then use it later
        """
        positive_feed = self.textEdit_possitive_feedback_interval.toPlainText()
        negative_feed = self.textEdit_negative_feedback_interval.toPlainText()
        # First condition --> check that the text boxes are not empty
        # TODO : Throw a warning box to enter something on either of the box

        # check user input
        # if(positive_feed_start == "" or positive_feed_end == "" or negative_feed_start == "" or negative_feed_end == ""):
        #     print("Please fill all the text boxes")
        #     pass
        # else:
        # NOTE: Condition if the size of either of the variables is not two raise an error
        # TODO : Throw a warning box to enter the time intervals in the correct format
        # if(len(positive_feed_start)!=2 or len(positive_feed_end)!=2 or len(negative_feed_start)!=2 or len(negative_feed_end)!=2):
        #     print("Please enter the time intervals in the correct format (MM:SS)")
        #     pass

        f_ptr = open(self.normal_summary_path, 'r')
        summary_one_hot = []
        for i in f_ptr:
            summary_one_hot.append(i[:-1])
        idxes = [idx for idx, val in enumerate(summary_one_hot) if val == '1']
        #print (idxes)
        positive_feed_list = positive_feed.split('\n')
        negative_feed_list = negative_feed.split('\n')
        positive_feedback_idxes = secs_to_indexes(positive_feed_list, idxes)
        negative_feedback_idxes = secs_to_indexes(negative_feed_list, idxes)
        positive_feedback_idxes_to_vid = list([val for idx, val in enumerate(idxes) if idx in positive_feedback_idxes])
        negative_feedback_idxes_to_vid = list([val for idx, val in enumerate(idxes) if idx in negative_feedback_idxes])

        customized_summary_path = generate_summary_with_feedback(self.dataset, self.video_name, self.normal_summary_path, positive_feedback_idxes_to_vid, negative_feedback_idxes_to_vid)
        #customized_summary_path = 'output_summary_with_feedback/Alin_Day_1.mp4_Disney_policy_grad_summary_length_600_subshot_size_200_hidden_dim_256_summary_with_feedback_krishan.txt'
        generate_video_from_summary(customized_summary_path, self.dataset, self.video_name)
        plot_normal_and_custom_summary(self.normal_summary_path, customized_summary_path)
        print ('---------------Customized Summary Generated-------------------------')














        # normal_summary = '/media/enigma/f0762f3b-20d1-42a7-9fe1-60385c4a8a3e/video_summarization/actor_critic/code_interactive_summ/output_summary_without_feedback/'
        # write_feedback_path = '/media/enigma/f0762f3b-20d1-42a7-9fe1-60385c4a8a3e/video_summarization/actor_critic/code_interactive_summ/gui_feedback/'
        # files = os.listdir(normal_summary)
        # for my_file in files:
        #     if '.txt' in my_file:
        #         f_ptr = open(normal_summary + my_file, 'r')
        #         summary_one_hot = []
        #         for i in f_ptr:
        #             summary_one_hot.append(i[:-1])
        #         idxes = [idx for idx, val in enumerate(summary_one_hot) if val == '1']
        #         print (idxes)
        #         positive_feed_list = positive_feed.split('\n')
        #         negative_feed_list = negative_feed.split('\n')
        #         positive_feedback_idxes = secs_to_indexes(positive_feed_list, idxes)
        #         negative_feedback_idxes = secs_to_indexes(negative_feed_list, idxes)
        #         write_feedback_indexes(write_feedback_path + 'positive_feedback.txt', list([val for idx, val in enumerate(idxes) if idx in positive_feedback_idxes]))
        #         write_feedback_indexes(write_feedback_path + 'negative_feedback.txt', list([val for idx, val in enumerate(idxes) if idx in negative_feedback_idxes]))
        # call_python_function_feedback = 'python gui_main_policy_long_vid_interactive_with_feedback.py -d datasets/'+ dataset_global +'_features.h5 -s datasets/'+ dataset_global +'_splits.json -m summe --gpu 0 --save-dir log/' + video_name_global + '_int_pol-split0 --split-id 0 --verbose'
        # #os.system(call_python_function_feedback)
        # temp_py_script = 'python generate_video_from_summary.py -d '+ dataset_global +' -v '+ video_name_global +' -s feedback'
        # os.system(temp_py_script)
        # print ('---------------Customized Summary Generated-------------------------')

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
