# -*- coding: utf-8 -*-
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

import os
import copy
import pywt




if __name__ == '__main__':   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    train_loader = torch.utils.data.DataLoader(
        datasets.GTSRB(
            "./data",
            split="train",
            download=True,
            transform=transforms.Compose([
                        transforms.Resize([224, 224]),
                        transforms.ToTensor()]),
        ),
        batch_size=128,
        shuffle=True, 
        num_workers=2,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.GTSRB(
            "./data",
            split="test",
            download=True,
            transform=transforms.Compose([
                        transforms.Resize([224, 224]),
                        transforms.ToTensor()]),
        ),
        batch_size=1,
        shuffle=True,
    )



    

    plt.figure(figsize = (4,5))
    
   
    
    pathToFigure = 'Figure'
    if not os.path.isdir(pathToFigure):
        os.mkdir(pathToFigure)

    fig=plt.figure(figsize = (5,10))
    
    
    
    titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
    titles_LH = ['LL', ' HH']
    classes=40
    train_LL = copy.deepcopy(train_loader)
    train_HH = copy.deepcopy(train_loader)

    for i in range(10):
        #for batch_idx, (inputs,targets) in enumerate(test_loader):
        #print("loader samples=",train_loader.dataset._samples[0])
        #print("data:",type(train_LL.dataset._samples[0]))
        for idx in range(0,len(train_loader.dataset._samples)):
            #if(targets == i):
                #LL, (LH, HL, HH) = pywt.dwt2(inputs[0,:,:,:],'haar')#inputs(batch,3,224,224)
                
                LL, (LH, HL, HH) = pywt.dwt2(train_LL.dataset[idx][0],'haar')#inputs(batch,3,224,224)
                
                # print("LL shape = ",LL.shape[0])
                # print("LL type= ",type(LL))
                HH_inv=pywt.idwt2((np.zeros((LL.shape)), 
                                   (np.zeros((LH.shape)), np.zeros((HL.shape)), HH)
                                   ),'haar')
                LL_inv=pywt.idwt2((LL, 
                                   (np.zeros((LH.shape)), np.zeros((HL.shape)), np.zeros((HH.shape)))
                                   ),'haar')
                

                #print (torch.min(inputs[0,:,:,:]),torch.max(inputs[0,:,:,:]))
                #print (LH.min(),LH.max(),len(LH))
                for id,a in enumerate([LL, LH, HL, HH]):  #3,114,114
                    ax = fig.add_subplot(2, 4, id + 1)
                    a=(a-a.min())/(a.max()-a.min())
                    ax.imshow(np.transpose(a,(1,2,0)), interpolation="nearest",vmin=0,vmax=1)
                    ax.set_title(titles[id], fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                for id,a in enumerate([LL_inv,HH_inv]):
                    ax = fig.add_subplot(2, 4, id + 1 + 4)
                    
                    a=(a-a.min())/(a.max()-a.min())
                    ax.imshow(np.transpose(a,(1,2,0)), interpolation="nearest",vmin=0,vmax=1)
                    ax.set_title(titles_LH[id], fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                # ax = fig.add_subplot(2, 4, id + 1 + 4)
                # print (HH.min(),HH.max(),len(HH))
                # ax.imshow(np.transpose(HH,(1,2,0)), interpolation="nearest",vmin=0,vmax=1)
                # ax.set_title(titles[id], fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.tight_layout()
                plt.show()

                # inputs, targets = inputs.to(device), targets.to(device)
                # inputs.requires_grad = True
                # outputs = model(inputs)
                # outputs,o_beforeSoft = model(inputs)
                # _, predicted = outputs.max(1)

                
                break
        break    

    save_path = os.path.join(pathToFigure, f'wavletData.jpg')
    plt.savefig(save_path)
    fig_ori=plt.figure(2)
    # save_path = os.path.join(pathToFigure, f'OriData.jpg')
    # a=np.transpose(inputs[0,:,:,:].detach().numpy(),(1,2,0))
    # ax_ori=fig_ori.add_subplot(1,1,1)
    # ax_ori.imshow(a)
    # ax_ori.set_title('Ori_data')
    # ax_ori.set_xticks([])
    # ax_ori.set_yticks([])
    # plt.savefig(save_path)
    