'''
author: lin.xu and huang.rong
date: 2021.2.12
information: be used to create EmotionLabel.exe
'''
import sys
import os
import shutil
import re

from PyQt5 import QtWidgets
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from functools import partial
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import get_label_demo


def copyfile(srcfile, dstfile):
    '''
    copy files
    :param srcfile: source file
    :param dstfile: destine file
    :return:None
    '''
    if not os.path.isfile(srcfile):
        print("%s not exist!" % srcfile)
    else:
        f_path, f_name = os.path.split(dstfile)
        if not os.path.exists(f_path):
            os.makedirs(f_path)
        shutil.copyfile(srcfile, dstfile)
        print("copy %s -> %s" % (srcfile, dstfile))

class picture(QWidget):
    '''
    create emotion label gui
    '''
    def __init__(self, classlist):
        super(picture, self).__init__()

        self.resize(1000, 1000)
        self.setWindowTitle("Video Annotation")

        #play videos
        self.player = QMediaPlayer(self)
        self.mVideoWin = QVideoWidget(self)

        self.mVideoWin = QVideoWidget(self)
        self.mVideoWin.setFixedSize(800, 800)
        self.mVideoWin.move(100, 150)
        self.player.setVideoOutput(self.mVideoWin)
        layout=QVBoxLayout()

        #initialize index
        self.index = 0
        #self.index = int(sys.argv[1])  #set the preliminary index of the label image
        self.maxindex = 0
        self.index_be = -1
        self.change_flag = 0

        self.timer = QTimer(self)
        self.timerflag = 0
        self.timer.timeout.connect(self.retime)
        #QPushButton.setClickable(True)

        # Import folder
        self.pic_path_list = []
        btn = QPushButton(self)
        btn.setText("Import folder")
        btn.move(20, 40)
        btn.clicked.connect(self.opendir)
        # print('self pic path is', self.pic_path_list)

        self.classlist = classlist

        # create label {positive/negative} button
        self.btn_list = []
        for i in range(len(self.classlist)):
            btn = QPushButton(self)
            btn.setText(self.classlist[i])
            btn.move(20 + 100 * i, 70)
            self.btn_list.append(btn)
            btn.clicked.connect(self.classify)#bind same event，according to event's sender to judge which button is pressed

        for id in range(len(self.btn_list)):
            self.btn_list[id].clicked.connect(partial(self.writelabel, self.btn_list[id].text(), self.btn_list, id))

        btn = QPushButton(self)
        btn.setText("Play Video")
        btn.move(20+100, 40)
        btn.clicked.connect(self.showimage)

        btn = QPushButton(self)
        btn.setText("last")
        btn.move(20, 100)
        btn.clicked.connect(self.show_beforeimage)

        btn = QPushButton(self)
        btn.setText("replay")
        btn.move(20 + 200, 100)
        btn.clicked.connect(self.reshow)

        btn = QPushButton(self)
        btn.setText("next")
        btn.move(20 + 100, 100)
        btn.clicked.connect(self.show_nextimage)

        self.setLayout(layout)

    # Import folder where 360 degree images needed to label
    def opendir(self):
        self.file_path = QFileDialog.getExistingDirectory(self, "please choose save path...", "./")
        print("file path is: %s" % self.file_path)
        file_list = os.listdir(self.file_path)
        name_suffix_true = ['mp']

        self.img_list = []
        self.img_type = []
        self.img_name = []

        for i in file_list:
            name_suffix = re.findall("[a-z]+", i)
            if name_suffix == name_suffix_true:   #only show files which suffix=='mp'
                self.img_list.append(i)
                pic_path = self.file_path + '/' + i
                self.pic_path_list.append(pic_path)

        self.maxindex = len(self.pic_path_list) - 1

    #label 360 degree images
    def classify(self):

        self.idx = self.index
        print("now index is: %d" % self.idx)
        print("last index is: %d" % self.index_be)

        sender = self.sender()

        if self.img_list[self.idx] in self.img_name:
            self.change_flag = 1

            with open(self.file_path + '/' + "label information.txt", 'r', encoding='gbk')as f:
                self.lines = f.readlines()

            ge = [self.img_list[self.idx]]
            new = ''
            for line in self.lines:
                mode = True
                for i in ge:
                    if i in line:
                        mode = False
                        break
                if mode:
                    new += line

            with open(self.file_path + '/' + "label information.txt", 'w')as f:
                f.write(new)
            self.img_type[self.idx] = sender.text()
            print('relabel  %s to %s...' % (self.img_list[self.idx], self.img_type[self.idx]))

        if self.change_flag == 0:
            self.img_type.append(sender.text())
            self.img_name.append(self.img_list[self.idx])
        print('present index video name is %s and its label is %s' % (self.img_name[-1],self.img_type[-1]))

        if (len(self.img_type))%10 == 0:
            QMessageBox.information(self, "message", "Congratulations you have finished tagging a set of panoramic videos! please take a rest... ", QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.Yes)
        self.change_flag = 0
        self.index_be = self.index


    def showimage(self):
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.pic_path_list[self.index])))
        self.player.play()

    def reshow(self):
        if self.index == self.maxindex:
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.pic_path_list[self.index-1])))
        else:
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.pic_path_list[self.index+1])))
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.pic_path_list[self.index])))
        self.player.play()

    def retime(self):
        self.timerflag = 1

    def show_nextimage(self):
        if self.timerflag == 1:
            self.timerflag = 0
            self.index += 1

            if self.index == len(self.pic_path_list):
                sys.exit()

            if self.index >= self.maxindex:
                self.maxindex = self.index

            for b_Button_id in range(len(self.btn_list)):
                self.btn_list[b_Button_id].setStyleSheet('background-color: #FFFFFF')

            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.pic_path_list[self.index])))
            self.player.play()
        else:
            self.timer.start(1000)  # ms level, set 10s means 100000ms
            QMessageBox.information(self, "Notes", "you need to watch at least 10s to ensure the validity of annotation...", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.pic_path_list[self.index])))
            self.player.play()

    def show_beforeimage(self):
        self.index -= 1
        for b_Button_id in range(len(self.btn_list)):
            self.btn_list[b_Button_id].setStyleSheet('background-color: #FFFFFF')

        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(self.pic_path_list[self.index])))
        self.player.play()

    def msg1(self):
        QMessageBox.information(self, "Congratulation", "task accomplished~", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

    def writelabel(self, label, btn_list, id):
        now_txt_path = self.file_path + '/' + "label information.txt"
        with open(now_txt_path, "a") as f:
            f.write(self.img_list[self.idx])
            f.write(" ")
            f.write(label)
            f.write('\n')
            f.close()

        if self.index == len(self.pic_path_list)-1:
            QMessageBox.information(QtWidgets.QWidget(), 'Info Tip', "this is the last one...", QMessageBox.Yes)

        # Trigger button's color change
        for b_Button_id in range(len(btn_list)):
            if b_Button_id == id:
                btn_list[b_Button_id].setStyleSheet('background-color: #ffff00')
            else:
                btn_list[b_Button_id].setStyleSheet('background-color: #FFFFFF')# #FFFFFF
            #time.sleep(5)
            #btn_list[b_Button_id].setStyleSheet('background-color: #FFFFFF')

if __name__ == "__main__":
    list = ["negative", "positive"]
    #starttime = time.time()
    print('Labeling starts, please take it seriously.')
    print('Labeling starts, please take it seriously.')
    print('Labeling starts, please take it seriously.')
    app = QtWidgets.QApplication(sys.argv)
    my = picture(list)
    my.show()
    sys.exit(app.exec_())