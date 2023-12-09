import yaml
import torch
import torch.nn as nn
from models import get_model
from dataset import get_dataset,get_wavelet
from optimizer import get_optimizer
from checkpoint import save_checkpoint,load_checkpoint
from logger import get_logger
from common.hog import mask_hog
from common.utils import *

import os

import torchvision.models as models
import matplotlib.pyplot as plt
import random
import numpy as np
from torchsummary import summary
from thop import profile
from torchvision.transforms import v2
from torchvision.io import encode_jpeg,decode_jpeg
import datetime
import argparse

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_model(model):
    summary(model, (3, 224, 224))
    input = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(model, inputs=(input, ))
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"FLOPs: {flops}, Params: {params}")

# Function to parse arguments from a YAML file
def parse_args(file_path='./config.yaml'):
    with open(file_path, 'r') as file:
        args = yaml.safe_load(file)
    return args

# Function to initiate training
def train():   
    #args = parse_args()
    # logger = get_logger(args['train']['log_path'], args['train']['log_file'])
    for model_name in args['models']:
        train_single_model(args,model_name,train_dataloader,test_dataloader,logger)
    return

# Function to train a single model
def train_single_model(args,model_name,train_dataloader,test_dataloader,logger):
    #model = get_model(model_name)
    
    # optimizer,schedueler = get_optimizer(args['optimizer']['name'],
    #                                      model,
    #                                      args['optimizer']['lr'],
    #                                      args['optimizer']['weight_decay'])
    #criterion = nn.CrossEntropyLoss()
    epochs = args['train']['epochs']
    print_model(model)
    model, optimizer, schedueler, start_epoch= load_checkpoint(model,model_name,optimizer,schedueler,args['train']['checkpoint_path'],logger)
    acc = 0.0
    best_acc = 0.0
    lr_values = []
    for epoch in range(start_epoch,epochs):
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            lr_step = optimizer.param_groups[0]['lr']
            lr_values.append(lr_step)

            optimizer.step()
            schedueler.step()
            if batch_idx % args['train']['log_interval'] == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_dataloader.dataset),
                        100. * batch_idx / len(train_dataloader), loss.item()))
                
        # Validation and checkpoint saving based on validation accuracy
        # if batch_idx % args['train']['val_interval'] == 0:
        if epoch % args['train']['val_interval'] == 0:
            acc=validataion(args,model,test_dataloader,criterion,epoch,logger)
            if acc > best_acc:
                best_acc = acc
                save_checkpoint(args['train']['checkpoint_path'],model_name,model,optimizer,epoch,schedueler)

        # Saving checkpoint based on save interval
        # if batch_idx % args['train']['save_interval'] == 0:
        #         save_checkpoint(args['train']['checkpoint_path'],model_name,model,optimizer,epoch,schedueler)


    return

# Function for model validation
def validataion(args,model,test_loader,criterion,epoch,logger):
    model.eval()  
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad(): 
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)  

    acc = correct / total
    avg_loss = test_loss / len(test_loader)

    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        avg_loss, correct, total, 100. * acc))
    
    return acc


    
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
# def attack(data_loader,A_name,eps_v=0.015,filter=0,hog=0):

    
#     total = 0
#     correct = 0
#     correct_A =0
#     correct_A_LL = 0
#     correct_A_HH = 0
#     acc_A_LL = 200
#     acc_A_HH = 200
#     logger.info('Attack start'+model_name+ ' '+A_name)
#     # filter = 1
#     # hog = 0
#     name_hog_f_list=[]
#     correct_hog_f_list =[]
#     name_f_list=[]
#     correct_f_list=[]
#     # correct_class=[[] for i in range(43)]
#     for batch_idx, (data, target) in enumerate(data_loader):

#         data, target =data.to(device), target.to(device) 
#         data_for_attack=data.detach().clone()
#         data_for_attack.requires_grad = True
#         data.requires_grad = True
#         # print ("data range:",torch.min(data[0,:,:,:]),"-",torch.max(data[0,:,:,:]))
#         output= model(data)
#         loss =  criterion(output, target)
#         #print("OutputSize",output.size())
#         #print("loss=",loss)
#         #print("Output",output)
#         model.zero_grad() #zero all existing gradients
#         optimizer.zero_grad() #zero all existing gradients
#         loss.backward()
#         # loss.backward(retain_graph=True)
#         # print('after',data.grad.data)
#         # loss.backward()
#         _, pred = output.max(1) #normal data classified result
#         correct += pred.eq(target).sum()#.item()
#         # correct_class[target][0]+= pred.eq(target).sum()
#         # if batch_idx <10:
#         #     print("target=",target)
#         #     print("predict = ",pred)
#         total += target.size(0)
#         match A_name:

#             case 'FGSM':           
#                 data_A,noise = FGSM (data ,eps_v,data.grad.data)               
#             case 'iFGSM':
#                 data_A,noise = iFGSM (data_for_attack,target ,eps_v)
#             case _:
#                 print("no this kind of attack mode!!")
#         noise_LL,noise_HH = get_wavelet(noise)
#         data_A_noise_LL =normalize( data + noise_LL.to(device))
#         data_A_noise_HH =normalize( data + noise_HH.to(device))
#         # data_A_mean =torch.mean(data_A)
#         # data_A_mean = data_A_mean.unsqueeze(1).expand(-1, 3, -1, -1)
        
#         # ini=0
#         # output_A = model(data_A)       #沒用到mask
#         # _, pred_A = output_A.max(1)
#         # correct_A += pred_A.eq(target).sum()#.item()
#         if batch_idx==0:
#             fig=plt.figure()
#             noise_norm=normalize(noise)
#             pic_t=['data_r','data_g','data_b','noise_r','noir_g','noise_b']
#             for id in range(3):
#                 ax = fig.add_subplot(2,3, id+ 1)
#                 ax.imshow(data[0,id,:,:].cpu().detach().numpy(), interpolation="nearest",vmin=0,vmax=1,cmap='gray')
#                 ax.set_title(pic_t[id], fontsize=10)
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#                 ax = fig.add_subplot(2,3, id+ 4)
#                 ax.imshow(noise_norm[0,id,:,:].cpu().detach().numpy(), interpolation="nearest",vmin=0,vmax=1,cmap='gray')
#                 ax.set_title(pic_t[id+3], fontsize=10)
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#                 fig.tight_layout()
#             save_path = os.path.join(pathToFigure, f'{A_name}_data_noise_rgb.jpg')
#             plt.savefig(save_path)
#         ini=3
#         if batch_idx==0:
#             pic_t=['data_A','data_A_noise_LL','data_A_noise_HH','data','noise','noise_LL','noise_HH']
#             name_f_list=pic_t[:ini]
#             fig=plt.figure()
#         for id,a in enumerate([data_A,data_A_noise_LL,data_A_noise_HH,data,normalize(noise),normalize(noise_LL),normalize(noise_HH)]):  
#             if id<3:
#                 output = model(a.to(device))
#                 _, pred = output.max(1)
#             if batch_idx==0:
#                 if id<3:
#                     correct_f_list.append(pred.eq(target).sum())
#                 ax = fig.add_subplot(2,4, id+ 1)
#                 a = a[0,:,:,:].cpu().detach().numpy()
#                 ax.imshow(np.transpose(a,(1,2,0)), interpolation="nearest",vmin=0,vmax=1)
#                 ax.set_title(pic_t[id], fontsize=10)
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#                 fig.tight_layout()
#                 # plt.show()
                
                
#             else:
#                 if id<3:
#                     correct_f_list[id]+=pred.eq(target).sum()
#         if batch_idx==0:
#             save_path = os.path.join(pathToFigure, f'{A_name}_noise_filter.jpg')
#             plt.savefig(save_path)
#         # data_A=data_A_noise_LL
        
        
#         if filter==1:
            
#             data_A_LL,data_A_HH = get_wavelet(data_A)
#             data_A_LL,data_A_HH=normalize(data_A_LL.to(device)),normalize(data_A_HH.to(device))    #沒用到mask
#             # data_A_grb=data_A[:, [1, 2, 0], :, :]
#             # # print(data_A_grb.size())
#             # data_A_grb_LL,data_A_grb_HH = get_wavelet(data_A_grb)
#             # data_A_grb_LL,data_A_grb_HH = normalize(data_A_grb_LL.to(device)),normalize(data_A_grb_HH.to(device))
            
#             # blurer= v2.GaussianBlur(kernel_size=9, sigma=( 5.))
#             # data_A_blur = blurer(data_A)
#             # data_A_blur_LL,data_A_blur_HH = get_wavelet(data_A_blur)
#             # data_A_blur_LL,data_A_blur_HH = normalize(data_A_blur_LL.to(device)),normalize(data_A_blur_HH.to(device))
#             if hog ==0:
#                 data_A_meanRGB = torch.mean(data_A,dim=[2,3])
#                 data_A_mean=torch.mean(data_A_meanRGB,dim=1)
#                 data_A_meanRGB = data_A_meanRGB.unsqueeze(2).unsqueeze(2).expand(-1, -1,224,224)
#                 # print('data_A_meanRGB size',data_A_meanRGB.size(),' data_A_meanRGB= ',data_A_meanRGB[0,:,0:2,0:2]) 
#                 data_A_miti = data_A_mean/data_A_meanRGB*data_A
            
#                 data_A_jpeg =(decode_jpeg(encode_jpeg((data_A[0,:,:,:]*255).to(torch.uint8).cpu(),quality=30),device=device)/255.).to(torch.float).unsqueeze(0)
#             # print('data_A_jpeg type:',type(data_A_jpeg),'\n shape',data_A_jpeg.size())
            
#                 if batch_idx==0:
#                     logger.info("JPEG quality (30)")
#                     # name_f_list.extend(['data_A_grb','data_A_grb_LL','data_A_grb_HH',
#                     #                 'data_A_blur','data_A_blur_LL','data_A_blur_HH','data_A_LL','data_A_HH'])
#                     name_f_list.extend(['data_A_miti','data_A_jpeg','data_A_LL','data_A_HH'])
#                     fig=plt.figure()
#                 for id,a in enumerate([data_A_miti,data_A_jpeg,data_A_LL,data_A_HH]):
#                     # [data_A_grb,data_A_grb_LL,data_A_grb_HH,
#                     #                    data_A_blur,data_A_blur_LL,data_A_blur_HH,
#                     #                    data_A_LL,data_A_HH]) 
#                     output = model(a.to(device))
#                     _, pred = output.max(1)
#                     if batch_idx==0:
#                         correct_f_list.append(pred.eq(target).sum())
#                         ax = fig.add_subplot(3, 3, id+ 1)
#                         a = a[0,:,:,:].cpu().detach().numpy()
#                         ax.imshow(np.transpose(a,(1,2,0)), interpolation="nearest",vmin=0,vmax=1)
#                         ax.set_title(name_f_list[id+ini], fontsize=10)
#                         ax.set_xticks([])
#                         ax.set_yticks([])
#                         fig.tight_layout()
#                         # plt.show()
                        
#                     else:
#                         correct_f_list[id+ini]+=pred.eq(target).sum()
#                 if batch_idx==0:
#                     save_path = os.path.join(pathToFigure, f'defense.jpg')
#                     plt.savefig(save_path)
#                 # output_A_LL = model(data_A_LL.to(device) )
#                 # output_A_HH = model(data_A_HH.to(device) ) 
#                 # _, pred_A_LL = output_A_LL.max(1)
#                 # _, pred_A_HH = output_A_HH.max(1)


#                 # correct_A_LL += pred_A_LL.eq(target).sum()#.item()
#                 # correct_A_HH  += pred_A_HH.eq(target).sum()#.item()
#         if hog ==1 :
#             ini = 0
#             # data_hog = mask_hog(data_A)
#             hog_mask,hog_mask_rgb,img_hog_mask_rgb= mask_hog(data_A[0,:,:,:])
            
            
#             pic_all_ori = [data]
#             pic_all_ori_t = ['data']
#             if batch_idx==0:
#                 fig=plt.figure()
#                 for id,a in enumerate(pic_all_ori):  #3,114,114
                    
#                     ax = fig.add_subplot(5, 3, id+ 1)
#                     a = a[0,:,:,:].cpu().detach().numpy()
#                     ax.imshow(np.transpose(a,(1,2,0)), interpolation="nearest",vmin=0,vmax=1)
#                     ax.set_title(pic_all_ori_t[id], fontsize=10)
#                     ax.set_xticks([])
#                     ax.set_yticks([])
#             # hog_mask=normalize(hog_mask)
#             # hog_mask[hog_mask>0]=1
#             img_hog_mask_rgb = normalize(img_hog_mask_rgb) 
#             img_hog_mask_rgb = np.transpose(img_hog_mask_rgb,(2,0,1))
#             hog_mask_rgb =normalize(hog_mask_rgb) #224,224,3
#             hog_mask_rgb = np.transpose(hog_mask_rgb,(2,0,1))#3,224,224
#             # hog_mask_rgb_m=hog_mask_rgb.copy()
#             # hog_mask_rgb_m[hog_mask_rgb_m>0]=1
#             hog_mask_bar= 1-hog_mask_rgb
            
            
                
                
#             # hog_mask = np.array([hog_mask]*3)
#             pic_all=[ hog_mask_rgb,img_hog_mask_rgb]#, hog_mask_rgb, img_hog_mask_rgb]
#             pic_all_t=[ 'hog_mask_rgb','img_hog_mask_rgb']#, 'hog_mask_rgb', 'img_hog_mask_rgb']
#             for id,a in enumerate(pic_all):  
#                 output = model(torch.tensor(a).unsqueeze(0).to(device))
#                 _, pred = output.max(1)
#                 if batch_idx==0:
#                     correct_hog_f_list.append(pred.eq(target).sum())
#                     name_hog_f_list.append(pic_all_t[id])
#                 else:
#                     correct_hog_f_list[id]+=pred.eq(target).sum()
#                 ax = fig.add_subplot(5, 3, (id) + 2)
#             #   print(pic_all_t[id]," range: ",a.max(),"-",a.min())
                
#               # ax.imshow(np.transpose(a,(1,2,0)), interpolation="nearest",vmin=0,vmax=1)
#                 if a.shape[2]==3:
#                     ax.imshow(a,vmin=0,vmax=1)
#                 else:
#                   ax.imshow(np.transpose(a,(1,2,0)),vmin=0,vmax=1)
#                 ax.set_title(pic_all_t[id], fontsize=10)
#                 # print(pic_all_t[id],type(a))
#             #     print(pic_all_t[id],a.shape)
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#             ini = id+1
                
            
            
#             # ax = fig.add_subplot(len(pic_all)+2, 2, 2)
#             # ax.hist(a.flatten(), linewidth=0.5, edgecolor="white")
#             # ax.set_title(pic_all_t[id]+" hist", fontsize=10)
#             data_A_mask = hog_mask_rgb*data_A[0,:,:,:].cpu().detach().numpy()        #沒用到filter
#             data_A_bar_mask = hog_mask_bar*data_A[0,:,:,:].cpu().detach().numpy()
#             # print('data_A_mask shape = ',data_A_mask.shape)
#             # print('data_A_bar_mask shape = ',data_A_bar_mask.shape)
#             data_A_m1_LL,data_A_m1_HH =get_wavelet(data_A_mask)
#             data_A_m0_LL,data_A_m0_HH =get_wavelet(data_A_bar_mask)
            
#             m0_1_m1_LL = normalize(torch.tensor(data_A_bar_mask).unsqueeze(0)+data_A_m1_LL)
#             m0_1_m1_HH = normalize(torch.tensor(data_A_bar_mask).unsqueeze(0)+data_A_m1_HH)

#             m0_0_m1_1 = normalize(torch.tensor(data_A_mask) .unsqueeze(0))   #m0_0_m1_1
#             m0_0_m1_LL = normalize(data_A_m1_LL)  #m0_0_m1_LL
#             m0_0_m1_HH = normalize(data_A_m1_HH)  #m0_0_m1_HH

#             m0_LL_m1_1 = normalize(data_A_m0_LL+torch.tensor(data_A_mask).unsqueeze(0))
#             m0_LL_m1_LL = data_A_m1_LL+m0_0_m1_LL#data_A_LL
#             m0_LL_m1_HH= normalize(data_A_m0_LL+ data_A_m1_HH)
#             m0_HH_m1_1 = normalize(data_A_m0_HH+torch.tensor(data_A_mask).unsqueeze(0))
#             m0_HH_m1_LL= normalize(data_A_m0_HH + data_A_m1_LL)
#             m0_HH_m1_HH = data_A_m1_HH+data_A_m0_HH #data_A_HH
            
#             pic_all=[data_A,
#                      m0_1_m1_LL ,
#                      m0_1_m1_HH ,
#                      m0_0_m1_1  ,
#                      m0_0_m1_LL ,
#                      m0_0_m1_HH ,
#                      m0_LL_m1_1 ,
#                      m0_LL_m1_LL,
#                      m0_LL_m1_HH,
#                      m0_HH_m1_1 ,
#                      m0_HH_m1_LL,
#                      m0_HH_m1_HH]
#             pic_all_t=[ 'data_A',
#                         'm0_1_m1_LL ',
#                         'm0_1_m1_HH ',
#                         'Attack image with mask  ',
#                         'm0_0_m1_LL ',
#                         'm0_0_m1_HH ',
#                         'm0_LL_m1_1 ',
#                         'm0_LL_m1_LL',
#                         'm0_LL_m1_HH',
#                         'm0_HH_m1_1 ',
#                         'm0_HH_m1_LL',
#                         'm0_HH_m1_HH']
            
            
#             for id,a in enumerate(pic_all):
#                 if pic_all_t[id]!='data_A':
#                     a=a.to(device)
#                     # print(pic_all_t[id])
#                     output = model(a)
#                 _, pred = output.max(1)
#                 if batch_idx==0:
#                     if pic_all_t[id]!='data_A':
#                         correct_hog_f_list.append(pred.eq(target).sum())
#                         name_hog_f_list.append(pic_all_t[id])
#                     ax = fig.add_subplot(5, 3, (id) + 4)
#                     if torch.is_tensor(a):
#                         if a.device == 'cpu':
#                             a=a.detach().numpy()  
#                         else:
#                             a=a.cpu().detach().numpy()
#                         a=a[0,:,:,:]              #3,114,114
#                     # print(pic_all_t[id]," range: ",a.max(),"-",a.min())
#                     # ax.imshow(np.transpose(a,(1,2,0)), interpolation="nearest",vmin=0,vmax=1)
#                     ax.imshow(np.transpose(a,(1,2,0)),vmin=0,vmax=1)
#                     ax.set_title(pic_all_t[id], fontsize=10)
#                     # print(pic_all_t[id],type(a))
#                     # print(pic_all_t[id],a.shape)
#                     ax.set_xticks([])
#                     ax.set_yticks([])
#                 else:
#                     if pic_all_t[id]!='data_A':
#                         correct_hog_f_list[id+ini-1]+=pred.eq(target).sum()
                    
#             if batch_idx==0:
#                 fig.tight_layout()
#                 # plt.show()
#                 save_path = os.path.join(pathToFigure, f'hogData.jpg')
#                 plt.savefig(save_path)
#         # if batch_idx==round((len(data_loader)>>4)):
#         #     break
#         # break       
    
                

#     acc = 100.*correct/total
#     # acc_A = 100.*correct_A/total  
#     # print('The following hog part data_A is data_A_noise_LL')
#     strg=model_name+'Attack Finish data volume=({}/{})({:.2f}), eps= {} Acc=({:.2f}%) \n'.format(
#             batch_idx,len(data_loader),1.0*batch_idx/len(data_loader),eps_v, acc) 
      
#     for id,correct_ele in enumerate(correct_f_list): 
#             strg=strg+'acc_'+name_f_list[id]+' = ({:.2f}%)\n'.format(100.*correct_ele/total)
#     if filter ==1 and hog==1:  
#         # if hog ==0:
#             # acc_A_LL = 100.*correct_A_LL/total
#             # acc_A_HH = 100.*correct_A_HH/total
        
#             # strg=strg+'acc_A_LL =({:.2f}%), acc_A_HH =({:.2f}%)'.format(acc_A_LL,acc_A_HH)
            
#         # else:
#             #  strg=strg+model_name+' Attack Finish, eps= = {} Acc=({:.2f}%),'.format(eps_v, acc)
#         # strg=strg+'The following hog part data_A is data_A_noise_LL.\n'
#         for id,correct_ele in enumerate(correct_hog_f_list): 
#             strg=strg+'acc_'+name_hog_f_list[id]+' = ({:.2f}%)\n'.format(100.*correct_ele/total)
            
#     logger.info(strg)
    
        
#     return acc

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
        # # # if batch_idx==round((len(data_loader)>>4)):
        #     break
        # break       
    


    acc = 100.*correct/total
  
    strg=model_name+' Attack Finish data volume=({}/{})({:.2f}), eps= {} Acc=({:.2f}%) \n'.format(
            batch_idx,len(data_loader),1.0*batch_idx/len(data_loader),eps_v, acc) 
      
    for id,correct_ele in enumerate(correct_f_list): 
            strg=strg+'acc_'+name_f_list[id]+' = ({:.2f}%)\n'.format(100.*correct_ele/total)
    if hog==1:  
    
        for id,correct_ele in enumerate(correct_hog_f_list): 
            strg=strg+'acc_'+name_hog_f_list[id]+' = ({:.2f}%)\n'.format(100.*correct_ele/total)
            
    logger.info(strg)
    
        
    return acc

if __name__ == "__main__":
    gpuparser =  argparse.ArgumentParser()
    gpuparser.add_argument('-gpu',type=str,default='0',help='which gpus to use')
    gpuargs=gpuparser.parse_args()
    # Checking for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(gpuargs.gpu))
    print('gpu:',torch.cuda.current_device())
    args = parse_args()
    logger = get_logger(args['train']['log_path'], str(datetime.date.today())+ args['train']['log_file'])
    criterion = nn.CrossEntropyLoss()
    
    logger.info('Loading data start')
    train_dataloader,test_dataloader = get_dataset(batch_size=args['train']['batch_size'],
                             num_workers=args['train']['num_workers'])
    logger.info('Loading data finish')
    
    
    
    for model_name in args['models']:
        model = get_model(model_name).to(device)
        optimizer,schedueler = get_optimizer(args['optimizer']['name'],
                                         model,
                                         args['optimizer']['lr'],
                                         args['optimizer']['weight_decay'])
        # model_path = args['train']['checkpoint_path']+'/'+model_name+'.pth'
        # # if no model exist, do training
        # if not os.path.isfile(model_path):
        #     input()
        #     train_single_model(args,model_name,train_dataloader,test_dataloader,logger)
        # else:
        print_model(model)
        model, optimizer, schedueler, start_epoch= load_checkpoint(model,model_name,optimizer,schedueler,args['train']['checkpoint_path'],logger,device)
        
        #print("start_epoch",start_epoch)
        model.eval() 
        gradients = None
        
        
        
        
        pathToFigure = 'Figure'
        # pathToModel = 'Model'
        if not os.path.isdir(pathToFigure):
            os.mkdir(pathToFigure)
    
       
        # for eps in np.linspace(0.015, 0.1, num=5, endpoint=True):
        #     acc = attack(test_dataloader,'FGSM',filter=1,hog=1,eps_v=eps)
        acc = attack(test_dataloader,'FGSM',eps_v=0.015,filter='wavelet',hog=1)
        acc = attack(test_dataloader,'FGSM',eps_v=0.015,filter='jpeg',hog=1)
        # break
    