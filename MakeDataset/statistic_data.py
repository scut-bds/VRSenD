
# coding=utf-8
'''
author: huang.rong
data: 2021/03/23
function: used to de-duplicated data, check omit and repeat labels returned by each annotators
'''

import os

def statistic_data(source_dir, txt_path):
    '''
    de-duplicated data, check omit and repeat labels returned by each annotators
    :param source_dir: the task of the annotator
    :param txt_path: the annotator returned information.txt by EmotionLabel.exe
    :return: None
    '''
    person_id = source_dir.split('\\')[-1]   # the task named by annotator ID
    txt_name = person_id + '.txt'            #save correct label information.txt

    save_clear_label_path = r'./clean'       #save processed result
    if not os.path.exists(save_clear_label_path):
        os.mkdir(save_clear_label_path)

    print('**********************************************************')
    print(" %s annotator label result as follows: " % person_id)
    img_name_list = os.listdir(source_dir)
    img_num = len(img_name_list)
    print("the number of assigned label task is %s " % img_num)

    all_id_list = [i.split('.')[0] for i in img_name_list]  # all_id_list means all assingned images
    all_label_id_list = []          #all_label_id_list means all labeled images

    #read label information from returned txt
    with open(txt_path, 'r') as f:
        data = f.readlines()
        data.sort()
    print('the annotator returned label number is ', len(data))

    for i in data:
        i = i.strip('\n')
        label_id = i.split('.')[0]
        all_label_id_list.append(label_id)

    set_label_id_list = list(set(all_label_id_list))   # set_label_id_list means all unique labeled images
    set_data = list(set(data))

    #the number of omit and need to relabel 360 degree images = all - labeled
    all_id_list.sort()
    all_label_id_list.sort()
    id_not_in_source = [i for i in all_id_list if i not in all_label_id_list]
    print('the number of omit label images is ', len(id_not_in_source))
    print("you need to relabel 360 degree image names are %s" % id_not_in_source)

    #check for duplicate labels
    same_id_list = [i for i in all_label_id_list if i not in set_label_id_list] #same_id_list means the same ID list
    same_id_num = len(all_label_id_list) - len(set_label_id_list) #the number of the same ID
    if same_id_num != 0:
        print('there are %s 360 degree iamges given the same labels' % same_id_num) #给出相同标签的视频不需要给出ID

    #rewrite de-duplicated label information to txt
    save_txt_path = os.path.join(save_clear_label_path, txt_name) # de-duplicated label
    with open(save_txt_path, 'w') as f:
        f.writelines(set_data)
        f.close()
    print('**********************************************************')


source_dir = r'F:\EmotionLabel\all_to_videos'       # the task of the annotator
txt_path = r'C:\Users\lab-626\Desktop\lx.txt'  #the annotator returned information.txt by EmotionLabel.exe
#txt_path = r'./clean/19.txt'            #the processed result by statistic information.txt
statistic_data(source_dir, txt_path)