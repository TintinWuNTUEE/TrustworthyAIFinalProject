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
    classes=40

    for i in range(10):
        for batch_idx, (inputs,targets) in enumerate(test_loader):
            if(targets == i):
                LL, (LH, HL, HH) = pywt.dwt2(inputs[0,:,:,:],'haar')#inputs(batch,3,224,224)
                print (torch.min(inputs[0,:,:,:]),torch.max(inputs[0,:,:,:]))
                print (LH.min(),LH.max(),len(LH))
                for i,a in enumerate([LL, LH, HL, HH]):  #3,114,114
                    ax = fig.add_subplot(1, 4, i + 1)
                    a=(a-a.min())/(a.max()-a.min())
                    ax.imshow(np.transpose(a,(1,2,0)), interpolation="nearest",vmin=0,vmax=1)
                    ax.set_title(titles[i], fontsize=10)
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
    save_path = os.path.join(pathToFigure, f'OriData.jpg')
    a=np.transpose(inputs[0,:,:,:].detach().numpy(),(1,2,0))
    ax_ori=fig_ori.add_subplot(1,1,1)
    ax_ori.imshow(a)
    ax_ori.set_title('Ori_data')
    ax_ori.set_xticks([])
    ax_ori.set_yticks([])
    plt.savefig(save_path)
    