import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
import os
from PIL import Image
import random
def vis_res():

    #with open('/home/wangfeiyu/mypywork/CV/PVReid/answer_vehicleid.txt', 'r') as f:
    with open('/home/wangfeiyu/mypywork/CV/PVReid/answer_veri776_noRerank.txt', 'r') as f:
        all = f.readlines()
    lines = []
    special = ['0615_c009_00008140_0','0108_c005_00078200_0','0597_c009_00065420_0','0742_c008_00026360_0','0546_c019_00015605_0']
    for line in all:
        if line.strip().split(' ')[0] in special:
            lines.append(line)
    print('res_list:',lines)
    #random.shuffle(lines)

    fig = plt.figure(figsize=(20, 12))
    for i in range(5):
        line = lines[i]
        print('第{}行'.format(i),line)
        res_list = line.strip().split(' ')
        for j in range(11):
            ax = fig.add_subplot(5, 11, i*11+j+1)  # 第一行显示query图像

            if j == 0:
                img = Image.open(os.path.join('/home/wangfeiyu/mypywork/CV/MyVID/datas/VeRi/image_query',res_list[j]+'.jpg'))
                # img_resized = img.resize((288,288))
                # img = np.array(img_resized)
                #img = imread(img)
            else:
                img = Image.open(os.path.join('/home/wangfeiyu/mypywork/CV/MyVID/datas/VeRi/image_test', res_list[j] +'.jpg'))
            img_resized = img.resize((288, 288))
            img = np.array(img_resized)
            ax.imshow(img)
            if j != 0:
                print(res_list[j].split('_')[0],res_list[0].split('_')[0])
                if res_list[j].split('_')[0] == res_list[0].split('_')[0]:
                    for spine in ['top', 'bottom', 'left', 'right']:
                        ax.spines[spine].set_color('green')
                        ax.spines[spine].set_linewidth(2)
                else:
                    for spine in ['top', 'bottom', 'left', 'right']:
                        ax.spines[spine].set_color('red')
                        ax.spines[spine].set_linewidth(2)

            ax.set_xticks([])
            ax.set_yticks([])
            #ax.axis('off')
    fig.subplots_adjust(wspace=0.1, hspace=0.5)


    plt.show()

vis_res()