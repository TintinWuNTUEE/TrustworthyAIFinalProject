# -*- coding: utf-8 -*-
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from logger import get_logger
from utils.hog import mask_hog
import os
import copy
import pywt
from torchvision.transforms import v2
from torchvision.io import encode_jpeg,decode_jpeg
import datetime
#from utils import progress_bar
import argparse
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(56180, 9000)
        # self.fc1 = nn.Linear(0, 9000)
        self.fc2 = nn.Linear(9000, 43)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # 計算 fc1 的输入大小
        # fc1_in_features = x.size(1) * x.size(2) * x.size(3)

        # # 更新 fc1 的输入大小
        # self.fc1 = nn.Linear(fc1_in_features, 9000)
        x = x.view(-1, 56180)
        # x = x.view(-1, fc1_in_features)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        #inputs=torch.reshape(inputs,(-1,1,28,28))
        
        optimizer.zero_grad()
        outputs = model(inputs)
        #print(inputs.size(),targets.size(),outputs.size())
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    # progress_bar(epoch, len(trainloader_trig), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            # % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Train Epoch: {} [\tLoss: {:.6f} | Acc:{:.6f} %'.format(
                        epoch,  train_loss/total,100.*correct/total))
    return train_loss/(batch_idx+1), 100.*correct/total
def test(dataloader):
    
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f %% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    test_loss = test_loss/total
    logger.info('Test  [\tLoss: {:.6f} | Acc:{:.6f} %]'.format(
                          test_loss,acc))
    # Save checkpoint.
   
    return test_loss/total, acc
def get_wavelet(image):
        if torch.is_tensor(image):
            if image.device == 'cpu':
                LL, (LH, HL, HH) = pywt.dwt2(image.detach().numpy(),'haar')
            else:
                LL, (LH, HL, HH) = pywt.dwt2(image.cpu().detach().numpy(),'haar')
        else:
            LL, (LH, HL, HH) = pywt.dwt2(image,'haar')
        # if torch.cuda.is_available():
        #     LL, (LH, HL, HH) = pywt.dwt2(image.cpu().detach().numpy(),'haar')#inputs(batch,3,224,224)
        # else:
        #     LL, (LH, HL, HH) = pywt.dwt2(image.detach().numpy(),'haar')#inputs(batch,3,224,224)
        # print (torch.min(inputs[0,:,:,:]),torch.max(inputs[0,:,:,:]))
        # print (LH.min(),LH.max(),len(LH))
        # for i,a in enumerate([LL, LH, HL, HH]):  #3,114,114
        #     ax = fig.add_subplot(1, 4, i + 1)
            #  a=(a-a.min())/(a.max()-a.min())
        #     ax.imshow(np.transpose(a,(1,2,0)), interpolation="nearest",vmin=0,vmax=1)
        #     ax.set_title(titles[i], fontsize=10)
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        # fig.tight_layout()
        # plt.show()
        
        # LL = (LL-LL.min())/(LL.max()-LL.min())      #range[0,1]
        # HH = (HH-HH.min())/(HH.max()-HH.min())      #range[0,1]

        # print("LL ndim=",LL.ndim)
        if LL.ndim == 4 :
            LL = F.interpolate(torch.tensor(LL),mode='area',size=[224,224])
            HH = F.interpolate(torch.tensor(HH),mode='area',size=[224,224])
        else:
            LL = F.interpolate(torch.tensor(LL).unsqueeze(0),mode='area',size=[224,224])
            HH = F.interpolate(torch.tensor(HH).unsqueeze(0),mode='area',size=[224,224])
        # return torch.tensor(LL),torch.tensor(HH)
        return LL,HH
def FGSM (image, eps_v,data_grad):
    n=eps_v*data_grad.sign()
    # print(SignGrad.type())
    # print(SignGrad.size())
    # print(image.type())
    # print(image.size())
    attack_i = image + n
    attack_i = torch.clamp(attack_i,0,1) #limit the value between 0 and 1
    return attack_i,n  

def iFGSM(image,label, eps_v,num_iter=20):
    # alpha and num_iter can be decided by yourself, I just quick set a value
    # alpha = eps_v/num_iter/100 
    alpha=1/255   
    attack_i = image
    
    for i in range(num_iter):
        # print('i=',i)
        # attack_i.requires_grad = True
        attack_i= attack_i.detach().clone()
        attack_i.requires_grad = True
        loss =  criterion(model(attack_i.to(device)), label)
        model.zero_grad() #zero all existing gradients
        optimizer.zero_grad()
        
        loss.backward()
        # print('after',attack_i.grad.data)
        attack_i=attack_i+alpha*attack_i.grad.data.sign()
        
        # attack_i=torch.clamp(attack_i,image-eps_v,image+eps_v)
        attack_i = torch.max(torch.min(attack_i, image+eps_v), image-eps_v) # clip new x_adv back to [x-epsilon, x+epsilon]
    n=attack_i-image
    return attack_i,n
def normalize(img):
    img=(img-img.min())/(img.max()-img.min())
    return img
def pic(pic_l,pic_l_t,row,col,title,filename=None,cmap=0):
    fig=plt.figure()
    for id,a in enumerate(pic_l):  
        # print('subplot ',pic_l_t[id])
        ax = fig.add_subplot(row, col, id+1)
        if torch.is_tensor(a):
            if a.device == 'cpu':
                a=a.detach().numpy()
            else:
                a=a.cpu().detach().numpy()
        if a.ndim==4:
            a=a[0,:,:,:]
        if a.shape[2]==3:
            ax.imshow(a,vmin=0,vmax=1)
        else:
          ax.imshow(np.transpose(a,(1,2,0)),vmin=0,vmax=1)
        ax.set_title(pic_l_t[id], fontsize=10)
        # print(pic_all_t[id],type(a))
            #   print(pic_all_t[id],a.shape)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle(title)
    if filename==None:
        filename=title+'.jpg'
    fig.tight_layout()
    # plt.show()
    save_path = os.path.join(pathToFigure, filename)
    plt.savefig(save_path)
        
def attack(data_loader,A_name,eps_v=0.015,filter='wavelet',hog=0):

    
    total = 0
    correct = 0
    logger.info('Attack start'+model_name+ ' '+A_name)

    name_hog_f_list=[]
    correct_hog_f_list =[]
    name_f_list=[]
    correct_f_list=[]
    # correct_class=[[] for i in range(43)]
    for batch_idx, (data, target) in enumerate(data_loader):

        data, target =data.to(device), target.to(device) 
        data_for_attack=data.detach().clone()
        data_for_attack.requires_grad = True
        data.requires_grad = True
        output= model(data)
        loss =  criterion(output, target)

        model.zero_grad() #zero all existing gradients
        optimizer.zero_grad() #zero all existing gradients
        loss.backward()
        # loss.backward(retain_graph=True)

        _, pred = output.max(1) #normal data classified result
        correct += pred.eq(target).sum()#.item()

        total += target.size(0)
        match A_name:

            case 'FGSM':           
                data_A,noise = FGSM (data ,eps_v,data.grad.data)               
            case 'iFGSM':
                data_A,noise = iFGSM (data_for_attack,target ,eps_v)
            case _:
                print("no this kind of attack mode!!")
                return None
        noise_LL,noise_HH = get_wavelet(noise)
        data_A_noise_LL =normalize( data + noise_LL.to(device))
        data_A_noise_HH =normalize( data + noise_HH.to(device))

        if batch_idx==0:
            fig=plt.figure()
            noise_norm=normalize(noise)
            pic_t=['data_r','data_g','data_b','noise_r','noir_g','noise_b']
            for id in range(3):
                ax = fig.add_subplot(2,3, id+ 1)
                ax.imshow(data[0,id,:,:].cpu().detach().numpy(), interpolation="nearest",vmin=0,vmax=1,cmap='gray')
                ax.set_title(pic_t[id], fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                ax = fig.add_subplot(2,3, id+ 4)
                ax.imshow(noise_norm[0,id,:,:].cpu().detach().numpy(), interpolation="nearest",vmin=0,vmax=1,cmap='gray')
                ax.set_title(pic_t[id+3], fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.tight_layout()
            save_path = os.path.join(pathToFigure, f'{A_name}_data_noise_rgb.jpg')
            plt.savefig(save_path)
       
        
            pic_all=[data_A,data_A_noise_LL,data_A_noise_HH,data,normalize(noise),normalize(noise_LL),normalize(noise_HH)]
            pic_t=['data_A','data_A_noise_LL','data_A_noise_HH','data','noise','noise_LL','noise_HH']
            pic(pic_all,pic_t,2,4,title=A_name+'_noise_'+filter) 
            
            name_f_list=pic_t[:3]
        
        correct_f_ini=0   
        for id,a in enumerate([data_A,data_A_noise_LL,data_A_noise_HH]):  

            output = model(a.to(device))
            _, pred = output.max(1)
            if batch_idx==0:
                
                correct_f_list.append(pred.eq(target).sum())     #correct_f_list start
                
            else:
                
                correct_f_list[id]+=pred.eq(target).sum()
        correct_f_ini=correct_f_ini+id+1


        if hog==1:
            hog_mask= mask_hog(data_A[0,:,:,:]) 
            hog_mask_rgb = normalize(np.array([hog_mask]*3) )  #3,224,224
            hog_mask_bar= 1-hog_mask 
            #m1_1
            data_A_mask = hog_mask_rgb*data_A[0,:,:,:].cpu().detach().numpy()        #沒用到filter
            #m0_1
            data_A_bar_mask = hog_mask_bar*data_A[0,:,:,:].cpu().detach().numpy()


        match filter:
            
            case  'wavelet':

                data_A_LL,data_A_HH = get_wavelet(data_A)
                data_A_LL,data_A_HH=normalize(data_A_LL.to(device)),normalize(data_A_HH.to(device))    #沒用到mask
                if batch_idx==0:
                    pic_all=[data_A,data_A_LL,data_A_HH]
                    pic_t=['data_A','data_A_LL','data_A_HH']
                    pic(pic_all,pic_t,1,3,title=A_name+'_defense_'+filter) 
                    name_f_list.extend(['data_A_LL','data_A_HH'])
                    
                    print('correct_f_ini=',correct_f_ini)
                for id,a in enumerate([data_A_LL,data_A_HH]):
                    output = model(a.to(device))
                    _, pred = output.max(1)
                    if batch_idx==0:
                        correct_f_list.append(pred.eq(target).sum()) 
                    else:
                        correct_f_list[id+correct_f_ini]+=pred.eq(target).sum() #前面ini=3個
                correct_f_ini=correct_f_ini+id+1

                if hog==1:
                    ini = 0
                    
                    data_A_m1_LL,data_A_m1_HH =get_wavelet(data_A_mask)
                    data_A_m0_LL,data_A_m0_HH =get_wavelet(data_A_bar_mask)

                    m0_1_m1_LL = normalize(torch.tensor(data_A_bar_mask).unsqueeze(0)+data_A_m1_LL)
                    m0_1_m1_HH = normalize(torch.tensor(data_A_bar_mask).unsqueeze(0)+data_A_m1_HH)

                    m0_0_m1_1 = normalize(torch.tensor(data_A_mask) .unsqueeze(0))   #m0_0_m1_1
                    m0_0_m1_LL = normalize(data_A_m1_LL)  #m0_0_m1_LL
                    m0_0_m1_HH = normalize(data_A_m1_HH)  #m0_0_m1_HH

                    m0_LL_m1_1 = normalize(data_A_m0_LL+torch.tensor(data_A_mask).unsqueeze(0))
                    m0_LL_m1_LL = data_A_m1_LL+m0_0_m1_LL#data_A_LL
                    m0_LL_m1_HH= normalize(data_A_m0_LL+ data_A_m1_HH)
                    m0_HH_m1_1 = normalize(data_A_m0_HH+torch.tensor(data_A_mask).unsqueeze(0))
                    m0_HH_m1_LL= normalize(data_A_m0_HH + data_A_m1_LL)
                    m0_HH_m1_HH = data_A_m1_HH+data_A_m0_HH #data_A_HH

                    pic_all=[data,
                             hog_mask_rgb,
                             data_A,
                             m0_1_m1_LL ,
                             m0_1_m1_HH ,
                             m0_0_m1_1  ,
                             m0_0_m1_LL ,
                             m0_0_m1_HH ,
                             m0_LL_m1_1 ,
                             m0_LL_m1_LL,
                             m0_LL_m1_HH,
                             m0_HH_m1_1 ,
                             m0_HH_m1_LL,
                             m0_HH_m1_HH]
                    pic_all_t=[ 'data',
                                'hog_mask_rgb',
                                'data_A',
                                'm0_1_m1_LL ',
                                'm0_1_m1_HH ',
                                'Attack image with mask  ',
                                'm0_0_m1_LL ',
                                'm0_0_m1_HH ',
                                'm0_LL_m1_1 ',
                                'm0_LL_m1_LL',
                                'm0_LL_m1_HH',
                                'm0_HH_m1_1 ',
                                'm0_HH_m1_LL',
                                'm0_HH_m1_HH']
                    if batch_idx==0:
                        pic(pic_all,pic_all_t,5,3,title=A_name+'_defense_'+filter+'_hogData')

                                        
            case 'jpeg':
                data_A_jpeg =(decode_jpeg(encode_jpeg((data_A[0,:,:,:]*255).to(torch.uint8).cpu(),quality=30),device=device)/255.).to(torch.float).unsqueeze(0)
                if batch_idx==0:
                    logger.info("JPEG quality (30)")
                    pic_all=[data_A,data_A_jpeg]
                    pic_t=['data_A','data_A_jpeg']
                    pic(pic_all,pic_t,1,3,title=A_name+'_defense_'+filter) 
                    
                    name_f_list.extend(['data_A_jpeg'])
                for id,a in enumerate([data_A_jpeg]):
                    output = model(a.to(device))
                    _, pred = output.max(1)
                    if batch_idx==0:
                        correct_f_list.append(pred.eq(target).sum())
                    else:
                        correct_f_list[id+correct_f_ini]+=pred.eq(target).sum()
                correct_f_ini=correct_f_ini+id+1
                # pic(correct_f_list,name_f_list,3,3,title=A_name+'_defense_'+filter)
                if hog==1:
                    data_A_mask = torch.tensor(data_A_bar_mask) #tensor@cpu
                    data_A_bar_mask = torch.tensor(data_A_bar_mask) #tensor@cpu
                    data_A_m1_j=(decode_jpeg(encode_jpeg((data_A_mask*255).to(torch.uint8),quality=30),device=device)/255.).to(torch.float).unsqueeze(0)
                    data_A_m0_j=(decode_jpeg(encode_jpeg((data_A_bar_mask*255).to(torch.uint8),quality=30),device=device)/255.).to(torch.float).unsqueeze(0)
                    
                    data_A_mask=data_A_mask.unsqueeze(0).to(device)
                    data_A_bar_mask=data_A_bar_mask.unsqueeze(0).to(device)
                    m0_1_m1_j = normalize(data_A_bar_mask+data_A_m1_j)
                    m0_1_m1_0 = normalize(data_A_bar_mask)  

                    m0_0_m1_1 = normalize(data_A_mask)   #m0_0_m1_1
                    m0_0_m1_j = normalize(data_A_m1_j)   #m0_0_m1_1

                    m0_j_m1_0 = normalize(data_A_m0_j)
                    m0_j_m1_1 = normalize(data_A_m0_j+data_A_mask)
                    m0_j_m1_j = normalize(data_A_m0_j+data_A_m1_j)
                    pic_all=[data,
                             hog_mask_rgb,
                             data_A,
                             m0_1_m1_j,
                             m0_1_m1_0,
                             m0_0_m1_1,
                             m0_0_m1_j,
                             m0_j_m1_0,
                             m0_j_m1_1,
                             m0_j_m1_j]
                    pic_all_t=['data'         ,
                               'hog_mask_rgb',
                               'data_A',
                               'm0_1_m1_j',
                               'm0_1_m1_0',
                               'm0_0_m1_1',
                               'm0_0_m1_j',
                               'm0_j_m1_0',
                               'm0_j_m1_1',
                               'm0_j_m1_j']
                    if batch_idx==0:
                        pic(pic_all,pic_all_t,5,2,title=A_name+'_defense_'+filter+'_hogData')
                    
                    

            case _:
                ##########grb testing############# not good defense
                # data_A_grb=data_A[:, [1, 2, 0], :, :]
                # # print(data_A_grb.size())
                # data_A_grb_LL,data_A_grb_HH = get_wavelet(data_A_grb)
                # data_A_grb_LL,data_A_grb_HH = normalize(data_A_grb_LL.to(device)),normalize(data_A_grb_HH.to(device))
                # ########blur testing########## not good defense
                # blurer= v2.GaussianBlur(kernel_size=9, sigma=( 5.))
                # data_A_blur = blurer(data_A)
                # data_A_blur_LL,data_A_blur_HH = get_wavelet(data_A_blur)
                # data_A_blur_LL,data_A_blur_HH = normalize(data_A_blur_LL.to(device)),normalize(data_A_blur_HH.to(device))
                #################################
                print("")
                ##########RGB adjustment############# not good defense, this formula has some issue
                # data_A_meanRGB = torch.mean(data_A,dim=[2,3])
                # data_A_mean=torch.mean(data_A_meanRGB,dim=1)
                # data_A_meanRGB = data_A_meanRGB.unsqueeze(2).unsqueeze(2).expand(-1, -1,224,224)
                # data_A_miti = data_A_mean/data_A_meanRGB*data_A
        if hog==1:
            ini=3
            for id,a in enumerate(pic_all[ini:]):
                a=a.to(device)
                output = model(a)
                _, pred = output.max(1)
                if batch_idx==0:
                    correct_hog_f_list.append(pred.eq(target).sum())
                    name_hog_f_list.append(pic_all_t[id+ini])
                else:    
                    correct_hog_f_list[id]+=pred.eq(target).sum()
            
                    

        # if batch_idx==2:
        # # if batch_idx==round((len(data_loader)>>4)):
        #     break
        # break       
    


    acc = 100.*correct/total
  
    strg=model_name+'Attack Finish data volume=({}/{})({:.2f}), eps= {} Acc=({:.2f}%) \n'.format(
            batch_idx,len(data_loader),1.0*batch_idx/len(data_loader),eps_v, acc) 
      
    for id,correct_ele in enumerate(correct_f_list): 
            strg=strg+'acc_'+name_f_list[id]+' = ({:.2f}%)\n'.format(100.*correct_ele/total)
    if hog==1:  
    
        for id,correct_ele in enumerate(correct_hog_f_list): 
            strg=strg+'acc_'+name_hog_f_list[id]+' = ({:.2f}%)\n'.format(100.*correct_ele/total)
            
    logger.info(strg)
    
        
    return acc

if __name__ == '__main__':   
    parser =  argparse.ArgumentParser()
    parser.add_argument('-gpu',type=str,default='0',help='which gpus to use')
    args=parser.parse_args()

    
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu") 
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
    print('gpu:',torch.cuda.current_device())
    # input()
    logger = get_logger('./logs', str(datetime.date.today())+'train.log')
    
    model_name='GTSRB_simpleCNN.pt'
    model = Net().to(device)
    # input()
    model.eval()
    print(model)        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

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
            download=False,
            transform=transforms.Compose([
                        transforms.Resize([224, 224]),
                        transforms.ToTensor()]),
        ),
        batch_size=1,
        shuffle=True,
    )

    pathToFigure = 'Figure'

    if not os.path.isdir(pathToFigure):
        os.mkdir(pathToFigure)

    if not os.path.isfile(model_name):
        # start training
        max_epochs=100
        start_epoch = 0
        loss_train_list = []
        acc_train_list=[]
        best_acc = 0
        for epoch in range(start_epoch, start_epoch+max_epochs):
            loss_train,acc_train=train(epoch)
            loss_train_list.append(loss_train)
            acc_train_list.append(acc_train)
            if acc_train > best_acc:
                print('Saving..')
                # state = {
                #     'net': model.state_dict(),
                #     'acc': acc_train,
                #     'epoch': epoch,
                # }

                #torch.save(state, model_name)
                torch.save(model.state_dict(), model_name) #只存weight
                best_acc = acc_train
        logger.info('Train best Acc:{:.6f} %'.format(best_acc))
        print("Training finished!")

        
        # Plot Loss
        save_path = os.path.join(pathToFigure, f'loss')
        plt.figure(figsize = (9,6))
        plt.grid()
        plt.plot(loss_train_list, marker='o', linewidth=2.5)
        plt.xlabel("Epoch", fontsize=13)
        plt.ylabel("Loss", fontsize=13)
        plt.title('Traning Loss', fontsize=15)
        plt.savefig(save_path)

        # Plot Acc
        save_path = os.path.join(pathToFigure, f'accuracy')
        plt.figure(figsize = (9,6))
        plt.grid()
        plt.plot(acc_train_list, marker='o', linewidth=2.5)
        plt.xlabel("Epoch", fontsize=13)
        plt.ylabel("Accuracy (%)", fontsize=13)
        plt.title('Traning Accuracy', fontsize=15)
        plt.savefig(save_path)
    else:

        model.load_state_dict(torch.load(model_name))          #一般正常load model方式
    

   
    
    
    # 
    # for eps in np.linspace(0.015, 0.1, num=5, endpoint=True):
    #     acc = attack(test_loader,'FGSM',eps_v=eps,filter=1)
    acc = attack(test_loader,'FGSM',eps_v=0.015,filter='wavelet',hog=1)
    acc = attack(test_loader,'FGSM',eps_v=0.015,filter='jpeg',hog=1)
    # acc = attack(test_loader,'iFGSM',eps_v=0.015,filter=1,hog=1)
    """
    titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
    titles_LH = ['LL', ' HH']
    classes=40
    train_LL = copy.deepcopy(train_loader)
    train_HH = copy.deepcopy(train_loader)
    fig=plt.figure(figsize = (5,10))
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
    """
    