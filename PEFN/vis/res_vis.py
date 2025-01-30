# -*- coding: utf-8 -*-            
# @Author : wobhky
# @File : res_vis.py
# @Time : 2024/7/2 15:50
import os
import re
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.image import imread
from torch import nn
import torch
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from opt import opt
from data import Data
from loss import Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature,extract_feature_vehicleID
from utils.metrics1 import mean_ap, cmc, re_ranking
from model.PEFN import PEFN
from data import VeRi
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
class Main():
    def __init__(self, model, loss, data, start_epoch=-1, eval_size='_small'):

        '''
        This can be replaced with your structure.
        For convenience in using three datasets, I added 'all' to the data,
        allowing the selection of different querysets and testsets based on the size of the data
        self.queryset = data.all['query{}'.format(eval_size)]
        self.testset = data.all['gallery{}'.format(eval_size)]
        self.query_loader = data.all['query{}_loader'.format(eval_size)]
        self.test_loader = data.all['gallery{}_loader'.format(eval_size)]
        '''

        self.model = model.cuda()

    def evaluate(self):
        # print(self.queryset.pids)
        # print(len(self.queryset.pids))
        # print("----------------------------------------------------")
        # print(self.testset.pids)
        # print("----------------------------------------------------")
        # print(self.queryset.camids)
        # print(len(self.queryset.camids))
        # print("----------------------------------------------------")
        # print(self.testset.camids)
        self.model.eval()
        txt = open(opt2.save_path, "w")
        #print(self.queryset.imgs)
        print('extract features, this may take a few minutes')

        qf = extract_feature(self.model, tqdm(self.query_loader)).numpy()
        gf = extract_feature(self.model, tqdm(self.test_loader)).numpy()

        def rank(dist):
            r = cmc(dist, self.queryset.pids, self.testset.pids, self.queryset.camids, self.testset.camids,
                    separate_camera_set=True,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(dist, self.queryset.pids, self.testset.pids, self.queryset.camids, self.testset.camids)

            return r, m_ap

        #########################   re rank##########################
        # q_g_dist = np.dot(qf, np.transpose(gf))
        # q_q_dist = np.dot(qf, np.transpose(qf))
        # g_g_dist = np.dot(gf, np.transpose(gf))
        # distmat = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        #
        # r, m_ap = rank(distmat)
        # print('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
        #       .format(m_ap, r[0], r[2], r[4], r[9]))
        # #########################no re rank##########################
        # print(qf.shape)
        # print(gf.shape)
        dist = cdist(qf, gf)

        r, m_ap = rank(dist)

        print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

        #f = self.result(distmat,txt)
        #f = self.result_remove_same_camera(distmat, txt)
        f = self.result_remove_same_camera(dist, txt)
        #f = self.result(dist, txt)

    def result(self, distmat, txt):#'/home/wangfeiyu/mypywork/CV/MyVID/datas/VeRi/image_query/0482_c004_00053910_0.jpg'
        m, n = distmat.shape
        q_n = [f.split("/")[-1].split(".")[0] for f in self.queryset.imgs]
        #print('qn:',q_n)
        g_n = [f.split("/")[-1].split(".")[0] for f in self.testset.imgs]
        # print('gn:', g_n)
        indices = np.argsort(distmat, axis=1)
        top20 = indices[:, :20]
        print('*******begin generate the top20 result***********')
        for i in range(len(top20)):
            name = q_n[i] + " "
            for s in range(20):
                name += g_n[int(top20[i, s])] + ' '
            name += '\n'
            txt.write(name)
        print('#########generate over#############')
        return True

    def result_remove_same_camera(self, distmat, txt):
        m, n = distmat.shape
        q_n = [f.split("/")[-1].split(".")[0] for f in self.queryset.imgs]
        g_n = [f.split("/")[-1].split(".")[0] for f in self.testset.imgs]
        indices = np.argsort(distmat, axis=1)
        #top20 = indices[:, :20]
        for i in range(m):
            name = q_n[i] + " "
            for j in range(100):
                if g_n[int(indices[i, j])].split("_")[1] ==  q_n[i].split("_")[1]:
                    continue
                name += g_n[int(indices[i, j])] + ' '
                if name.count(' ') == 21:
                    #print('name:',name)
                    break
            name += '\n'
            txt.write(name)
        print('#########generate over#############')
        return True


if __name__ == '__main__':
    data = Data(data="veri")
    model = PEFN(data.nums)
    cudnn.benchmark = True
    loss = Loss()
    start_epoch=-1

    #main1 = Main(model, loss, data, start_epoch, 'small')
    #main2 = Main(model, loss, data,start_epoch,'medium')
    #main3 = Main(model, loss, data,start_epoch,'large')
    main = Main(model, loss, data, start_epoch, "")
    if opt2.mode == 'evaluate':
        print('start evaluate')
        model.load_state_dict(torch.load(opt2.weight))
        main.evaluate()
