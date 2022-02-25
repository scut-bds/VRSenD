# coding=utf-8
'''
author: huang.rong
date: 2021/3/23
information:
be used to give final labels and  it's confidence level according to the principle of the minority obeying the majority
and saved the final result to vote_result.xlsx
all_information.xlsx format as: id:range(0,6364)，filename: XXX.mp4，person:range(1,20)，label:range(0,1)
vote_result.xlsx format as: id: 0-1272; filename: XXX.mp4; all label; appear num; percent
filename2label.xlsx format as: id: 0-1273，filename: XXX.mp4，person ID:1-20，label:1or0

'''

import pandas as pd
from  pandas import DataFrame


def get_vote_label(label_list, label_num):
    '''
    give the final label and some useful information and save in vote_label.xlsx
    :param label_list: all annotators returned labels
    :param label_num: each 360 degree images need how many annotators
    :return:
    '''
    same_label_list = [label_list[i:i+label_num] for i in range(0, len(label_list), label_num)] #in units of label_num to get same images label
    unique_label_list = [max(same_label_list[i], key=same_label_list[i].count) for i in range(len(same_label_list))]#according to the principle of the minority obeying the majority
    appear_num_list = [same_label_list[i].count(unique_label_list[i]) for i in range(unique_data_num)]#the appear times of final labels
    print('the appear times of final labels %s' % appear_num_list)
    percent_list = [(str(i / label_num * 100) + '%') for i in appear_num_list] #the confidence of the final label
    print('the confidence of final labels %s' % percent_list)

    rename_filename_list = [i.split('.')[0] + '.jpg' for i in unique_filename_list]  # change .mp4 to .jpg
    dict_data = {'filename':rename_filename_list,
                 'final label':unique_label_list,
                 'all label':same_label_list,
                 'appear num':appear_num_list,
                 'percent':percent_list}
    DataFrame(dict_data).to_excel(save_result_path, sheet_name='result')


    key_data = {'filename': rename_filename_list,
                 'label': unique_label_list}
    DataFrame(key_data).to_excel(save_label_path, sheet_name='result')

    unique_data_dict = dict(zip(unique_filename_list, unique_label_list)) #the final vote result combine into a dict
    return unique_data_dict

def get_effective_rate(unique_data_dict):
    '''
    compute every annotator's effective rate
    :param unique_data_dict: the vote result, format like {filename:label}
    :return: None
    '''
    all_person_index = [[i for i,x in enumerate(person_list) if x == (k+1)] for k in range(person_num)] #every annotator's index on the total result
    all_filename_index = [[filename_list[i] for i in all_person_index[j]] for j in range(person_num)] #every annotator's assigned filename
    all_label_index = [[label_list[i] for i in all_person_index[j]] for j in range(person_num)]   #every annotator's label
    all_classify_list = [dict(zip(all_filename_index[i], all_label_index[i])) for i in range(person_num)] #every annotator's dict{filename:label}
    all_num_list = [len(all_person_index[i]) for i in range(len(all_person_index))] #the number of every annotators assigned task
    print('every person need to label the video list number  is:', all_num_list)

    all_true_num_list = []  # means every annotators label equal to final label times
    num = 0
    for i in range(len(all_classify_list)):
        for key, value in all_classify_list[i].items():
            if value == unique_data_dict[key]:
                num = num+1
        all_true_num_list.append(num)
        num = 0
    print('every person label correctly number is: ', all_true_num_list)
    effective_rate_list = ['{:.2f}%'.format(all_true_num_list[i]/all_num_list[i]*100) for i in range(len(all_person_index))]
    print('every person label effective rate is: ', effective_rate_list)



data_path = r'./information.xlsx'       #the excel need to analysis which saved all annotators label result
save_result_path = r'./vote_result2.xlsx'    #the excel to save voted relult{filename：label}
save_label_path = r'./filename2label2.xlsx'  #

#read information from all_information.xlsx
label_num = 5                                     #how many annotators required to label every 360 degree images
data = pd.read_excel(data_path, engine='openpyxl')
all_data_num = len(data)                          #total number
filename_list = data['filename'].values.tolist()  #filename
label_list = data['label'].values.tolist()        #label
person_list = data['person'].values.tolist()      #corresponding person
all_data_dict = dict(zip(filename_list,label_list))  #{filename:label}

# get the unique information
unique_filename_list = list(set(filename_list))
unique_filename_list.sort()
unique_person_list = list(set(person_list))
unique_person_list.sort()
person_num = len(unique_person_list)
unique_data_num = len(unique_filename_list)

unique_data_dict = get_vote_label(label_list, label_num)
get_effective_rate(unique_data_dict)