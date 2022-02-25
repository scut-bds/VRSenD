'''
author: luo.wanxiang
date: 2021/1/16
information: to divide the task of each annotator according to the requirements
'''

import pandas as pd
import os
import random
import shutil


def randomdiv(m, n, banlist, primelist):
    '''
    n numbers are printed randomly from 1~m
    :param m: amount
    :param n: number
    :param banlist: list to avoid the same
    :param primelist: list to firstly assign
    :return: None
    '''

    results = []
    if (len(banlist) + n) > m:
        print('we are understaffed...please increase the manpower...')
        return
    for i in banlist:       #make sure banlist and primelist are not duplicated
        for j in primelist:
            if i == j:
                primelist.remove(j)
                print(r'banlist and primelist have same elements:"', j, r'", deleted')
    while len(results) < n:
        if primelist:
            random.shuffle(primelist)   #ramdom sort primelist
            s=primelist.pop()

        else:
            s = random.randint(1, m)

        if s not in banlist:
            results.append(s)   #selected annotators
            banlist.append(s)   #append selected annotators to banlist to aviod choose repeatedly

    print('allocate results: ', results)
    print('banlist: ', banlist)
    print('when exiting randomdiv function, primelist is ', primelist)
    return results, banlist


def update_primelist(list):
    nmin = len(list)
    nminlist = []
    for i in list:
        n = list.count(i)
        if n <= nmin:
            if not i in nminlist:
                nmin = n
                nminlist.append(i)
    return nminlist


def task_div(m, n, path):
    '''
    divide the task of each annotator according to the requirements
    :param m: the number of annotators
    :param n: the number of annotators to label each 360 degree images
    :param path: directory saved total 360 degree images to label
    :return: None
    '''
    for i in range(1, m+1):
        if not os.path.isdir(os.path.join(path, str(i))):
            os.makedirs(os.path.join(path, str(i)))
    picture_list = []
    user = []
    files = os.listdir(path)
    vlist = []
    i: int      #control the video index
    jnmax = 0   #the max number of 360 degree images in the same video

    for file in files:          #make sure the annotators can finish the label task
        if file.endswith('.jpg'):
            mixname, suffix = os.path.splitext(file)
            vname, pname = mixname.split('_')
            if not vname in vlist:
                jn = 1
                vlist.append(vname)
            else:
                jn = jn + 1
                if jn > jnmax:
                    jnmax = jn
                    maxname = vname
    if jn * n > m:
        print('the number of annotators is not enough... can not finish the task...')
        print('we need at least ', jnmax, '*', n, '=', jnmax * n, ' annotators...')
        print('the number of 360 degree images is too much... ', maxname)
        return

    totalresult = []
    for x in range(1,m):    #lead entry once in order tp upgrade primelist correctly
        totalresult.append(x)
    vlist = []    # reset vlist
    banlist = []  #aviod duplication
    primelist = []  #priority assigned annotators

    for file in files:
        if file.endswith('.jpg'):
            print('filename is ',file)
            mixname, suffix = os.path.splitext(file)
            vname, pname = mixname.split('_')
            print('video index is ',vname)
            if not vname in vlist:  #调整i控制视频号；如果不是同一视频；
                banlist = []        #初始化banlist
                if not banlist:
                    print('reset banlist...')
                if not vlist:
                    i = 0
                else:
                    i = i + 1
                vlist.append(vname)
                print(vlist)
                print('now index is ', i)
                divres, banlist = randomdiv(m, n, [], primelist)

            else:  #if this 360 degree image came from the same video
                if banlist:
                    divres, banlist = randomdiv(m, n, banlist, primelist)

            for item in divres:
                totalresult.append(item)

            primelist = update_primelist(totalresult)

            for i in range(n):      #add n times of file to picture_list by needed labels
                picture_list.append(file)
                user.append(divres[i])
                shutil.copyfile(os.path.join(path, file), os.path.join(path, str(divres[i]), file))


    Piclist = pd.DataFrame(picture_list)
    Userlist = pd.DataFrame(user)
    Piclist.columns = ['image name']
    Userlist.columns = ['annotators']

    data = pd.DataFrame
    data = Piclist.join(Userlist)
    data.to_excel(os.path.join(path, 'Task_Div.xlsx'))


path = './result/'
task_div(20, 5, path)




