import os
import re
import time
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
import torch
from torch.optim import lr_scheduler
from opt import opt
from datetime import datetime
from data import Data
import random
from loss.loss import Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature
from utils.metrics import euclidean_dist,re_rank,getRank
from model.PEFN import PEFN

#Vehicel Re-identification
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True,warn_only=True)


os.environ['CUDA_VISIBLE_DEVICES'] = "1"
#Additional models can be extended here
# to facilitate result comparison and ablation experiments
Model = {'PEFN':PEFN}
class DoTraing():
    def __init__(self, model, loss, data,start_epoch=-1):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        self.queryset = data.queryset
        self.testset = data.testset
        self.model = model.cuda()
        self.loss = loss
        self.flag=True
        self.start_epoch = start_epoch
        self.optimizer = get_optimizer(model)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1)

    def train(self):
        if self.start_epoch != -1 and self.flag:
            self.flag = False
            for i in range(self.start_epoch):
                self.scheduler.step()
        else:
            self.scheduler.step()
        self.model.train()
        print('start training')
        sum_loss = 0
        for batch, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            sum_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        print('batch:',batch)
        print('\nsum_loss:{}'.format(sum_loss))

    def evaluate(self):
        
        self.model.eval()
        print('extract features, this may take a few minutes')
        qf = extract_feature(self.model, tqdm(self.query_loader))
        gf = extract_feature(self.model, tqdm(self.test_loader))
        ######################### no re rank##########################
        distmat = euclidean_dist(qf, gf)
        cmc, mAP = getRank(distmat,self.queryset.pids,self.testset.pids,self.queryset.camids,self.testset.camids)
        print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(mAP, cmc[0], cmc[2], cmc[4], cmc[9]))
        #########################re rank##########################
        # distmat = re_rank(qf, gf)
        # cmc, mAP = getRank(distmat,self.queryset.pids,self.testset.pids,self.queryset.camids,self.testset.camids)
        # #cmc, mAP = getRank(distmat, self.query_.pids, self.valset.pids, self.query_.camids, self.valset.camids)
        # print('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
        #       .format(mAP, cmc[0], cmc[2], cmc[4], cmc[9]))

def Main(data, model, loss):
    print("Class:",data.nums)
    if opt.ngpu>1:
        model=nn.DataParallel(model,device_ids=[0, 1])
    start_epoch=-1
    if opt.resume and os.path.exists(opt.weight):
        start_epoch=int(re.sub("\D", "",opt.weight.split("/")[-1]))
        model.load_state_dict(torch.load(opt.weight))
    main = DoTraing(model, loss, data, start_epoch)
    if opt.mode == 'train':
        modelName = str(type(model)).split('.')[-1][:-2]
        print(modelName,"Training")
        for epoch in range(1+start_epoch, opt.epoch + 1):
            st = datetime.now()
            print('\nepoch', epoch)
            main.train()
            end = 1.0*(datetime.now()-st).seconds
            print('\nTime used:',round(end/60,2))
            if epoch > 100 and epoch % 10 == 0:
                savePath = 'yourlike/' + modelName
                os.makedirs(savePath, exist_ok=True)
                torch.save(model.state_dict(), (savePath + '/model_{}.pt'.format(epoch)))
    if opt.mode == 'evaluate':
        print('start evaluate')
        model.load_state_dict(torch.load(opt.weight))
        main.evaluate()


if __name__ == '__main__':

    num_gpus = torch.cuda.device_count()
    print("Number of available GPUs: ", num_gpus)
    dtype = opt.dtype
    if dtype == 'vehicle':
        size = (288, 288)
    else:
        size = (384, 128)
    set_seed(42)
    data = Data(data=opt.data_name,size=size)
    start = time.time()
    model = Model[opt.name](data.nums)#data.nums denotes datasets.len
    loss = Loss(margin=1.2)
    Main(data, model, loss)
    print('Time used:',round((time.time()-start)/60,2))



