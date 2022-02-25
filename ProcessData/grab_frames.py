'''
author: huang.rong
date: 2020.12.20
'''

import os
import  numpy as np
import cv2
from skimage.measure import compare_ssim
import time
import pandas as pd


def grab_frames(root, name, num, threshold, path):
    '''
    capture key frames of videos
    :param root: video path like './VR-HM48'
    :param name: video name like '1_48_070.mp4'
    :param num: the number of frames captured in the video
    :param threshold: threshold of the ssim value
    :param path: video path like './dataset'
    :return: None
    '''

    cap = cv2.VideoCapture(os.path.join(root, name))
    isOpened = cap.isOpened                 # check if the video is playing
    save_path = os.path.join(path, 'result')# directory to save the captured frames
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # get same video information
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   #the total frames of the video
    print('the total frames of the video is %d' % total_frame)
    fps = int(cap.get(cv2.CAP_PROP_FPS))                # frame rate
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))      # width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    # height
    print('frame rate, width and height of the video is %d, %d, %d' % (fps, width, height))
    frame_list = []   #captured frames list
    i = 1         #the index of captured frames
    abandon_frames = fps*15   #15s
    part = (total_frame - 2*abandon_frames)/num   #abandon the start and end 15s to avoid captured caption, and divide into num copies
    if part < 0:
        part = total_frame/num   #if the video is short, we can divide in to num copies directly
        current_frame = 0
    fname, _ = os.path.splitext(name)

    # start to capture frames
    while isOpened:
        if current_frame >= total_frame or i > num:
            break
        else:
            filename = fname + '_' + str(i).zfill(4) + '.jpg'  # name the captured frame
            ret, frame = cap.read()  #the current frames
            if ret:
                cv2.imencode('.jpg', frame)[1].tofile(os.path.join(save_path, filename))
                frame_list.append(os.path.join(save_path, filename))
                current_frame += i*part
                i += 1
            else:
                break
    # compute the similarity comparison
    img = cv2.imdecode(np.fromfile(frame_list[0], dtype=np.uint8), -1)
    img1 = cv2.imdecode(np.fromfile(frame_list[1], dtype=np.uint8), -1)
    img2 = cv2.imdecode(np.fromfile(frame_list[2], dtype=np.uint8), -1)
    ssim1 = compare_ssim(img, img1, multichannel=True)
    ssim2 = compare_ssim(img1, img2, multichannel=True)
    ssim3 = compare_ssim(img, img2, multichannel=True)
    if ssim1 > threshold:
        os.remove(frame_list[1])
    if ssim2 > threshold:
        os.remove(frame_list[2])
    if ssim3 > threshold:
        os.remove(frame_list[0])
    print('div ' + name + ' done')
    return


if __name__ == '__main__':

    path = r'J:\VR\dataset\1_60'
    time_start = time.time()
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:    #used to handle dataset videos
            if file.endswith('.mp4'):
                grab_frames(path, file, 5, 0.8, path)

        for name in files:    #used to handle bilibili videos
            if name.endswith('.mp4'):
                grab_frames(root, name, 5, 0.8, path)
    time_end = time.time()
    print('time cost ', time_end-time_start, 's')

