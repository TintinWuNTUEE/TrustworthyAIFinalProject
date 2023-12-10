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
    # print(model)
    # print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    # print(f"FLOPs: {flops}, Params: {params}")

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

    

def attack(data_loader,A_name,eps_v=0.015,filter='wavelet',mask=0): #train.py
#mask:0,none;1,hog;2,rembg
    mask_l=['hog','rembg']
    total = 0
    correct = 0
    information='Attack start '+model_name+ ' '+A_name+ ' e '+str(eps_v)+' defense '+filter
    if mask>0:
        information=information+' with '+mask_l[mask-1]
    logger.info(information)

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

        if batch_idx==0 :
            # print('attack rgb analysis plot start')
            if eps_v>0:
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
                    plt.suptitle(f'{model_name}_{A_name}_noise_{eps_v}_rgb')
                    fig.tight_layout()
                save_path = os.path.join(pathToFigure, f'{model_name}_{A_name}_noise_{eps_v}_rgb.jpg')
                plt.savefig(save_path)
            pic_all=[data_A,data_A_noise_LL,data_A_noise_HH,data,normalize(noise),normalize(noise_LL),normalize(noise_HH)]
            pic_t=['data_A','data_A_noise_LL','data_A_noise_HH','data','noise','noise_LL','noise_HH']
            if eps_v>0:
                pic(pic_all,pic_t,2,4,title=model_name+'_'+A_name+'_noise_'+str(eps_v)) 
            # print('attack rgb analysis plot finish')
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


        # if mask==1: #hog
        if mask==1: #hog
            hog_mask= mask_hog(data_A[0,:,:,:]) 
            hog_mask_rgb = normalize(np.array([hog_mask]*3) )  #3,224,224
            hog_mask_bar= 1-hog_mask 
            #m1_1
            data_A_mask = hog_mask_rgb*data_A[0,:,:,:].cpu().detach().numpy()        #沒用到filter
            #m0_1
            data_A_bar_mask = hog_mask_bar*data_A[0,:,:,:].cpu().detach().numpy()
            # print('data_A_bar_mask type:',torch.tensor(data_A_bar_mask))
            # print('hog_mask type: ',hog_mask.dtype)
        elif mask==2:#rembg
            
            data_A_mask = get_targetArea(data_A[0,:,:,:]) #3,224,224
            if torch.cuda.is_available():
                data_A_bar_mask = data_A[0,:,:,:].cpu().detach().numpy()-data_A_mask
            else:
                data_A_bar_mask = data_A[0,:,:,:].numpy()-data_A_mask
            
            hog_mask_rgb = data_A_mask.copy()
            hog_mask_rgb[data_A_mask>0]=1 
        match filter:
            
            case  'wavelet':

                data_A_LL,data_A_HH = get_wavelet(data_A)
                data_A_LL,data_A_HH=normalize(data_A_LL.to(device)),normalize(data_A_HH.to(device))    #沒用到mask
                if batch_idx==0:
                    if eps_v>0:
                        pic_all=[data_A,data_A_LL,data_A_HH]
                        pic_t=['data_A','data_A_LL','data_A_HH']
                    
                        pic(pic_all,pic_t,1,3,title=model_name+'_'+A_name+'_noise_'+str(eps_v)+'_defense_'+filter) 
                    name_f_list.extend(['data_A_LL','data_A_HH'])
                    
                    # print('correct_f_ini=',correct_f_ini)
                for id,a in enumerate([data_A_LL,data_A_HH]):
                    output = model(a.to(device))
                    _, pred = output.max(1)
                    if batch_idx==0:
                        correct_f_list.append(pred.eq(target).sum()) 
                    else:
                        correct_f_list[id+correct_f_ini]+=pred.eq(target).sum() #前面ini=3個
                correct_f_ini=correct_f_ini+id+1

                # if mask==1:
                if mask>0:
                    ini = 0
                    
                    data_A_m1_LL,data_A_m1_HH =get_wavelet(data_A_mask)
                    data_A_m0_LL,data_A_m0_HH =get_wavelet(data_A_bar_mask)

                    m0_1_m1_LL = normalize(torch.tensor(data_A_bar_mask).unsqueeze(0)+data_A_m1_LL)
                    m0_1_m1_HH = normalize(torch.tensor(data_A_bar_mask).unsqueeze(0)+data_A_m1_HH)

                    m0_0_m1_1 = normalize(torch.tensor(data_A_mask) .unsqueeze(0))   #m0_0_m1_1
                    m0_0_m1_LL = normalize(data_A_m1_LL)  #m0_0_m1_LL
                    m0_0_m1_HH = normalize(data_A_m1_HH)  #m0_0_m1_HH

                    m0_LL_m1_1 = normalize(data_A_m0_LL+torch.tensor(data_A_mask).unsqueeze(0))
                    m0_LL_m1_LL = normalize(data_A_m0_LL+data_A_m1_LL)#data_A_LL
                    m0_LL_m1_HH= normalize(data_A_m0_LL+ data_A_m1_HH)
                    m0_HH_m1_1 = normalize(data_A_m0_HH+torch.tensor(data_A_mask).unsqueeze(0))
                    m0_HH_m1_LL= normalize(data_A_m0_HH + data_A_m1_LL)
                    m0_HH_m1_HH = data_A_m1_HH+data_A_m0_HH #data_A_HH

                    pic_all=[data,
                             hog_mask_rgb,
                             data_A_mask,
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
                                'mask_rgb',
                                'data_A_mask',
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
                    if batch_idx==0 and eps_v>0:
                        pic(pic_all,pic_all_t,5,3,title=model_name+'_'+A_name+'_noise_'+str(eps_v)+'_defense_'+filter+'_'+mask_l[mask-1])

                                        
            case 'jpeg':
                data_A_jpeg =(decode_jpeg(encode_jpeg((data_A[0,:,:,:]*255).to(torch.uint8).cpu(),quality=30),device=device)/255.).to(torch.float).unsqueeze(0)
                if batch_idx==0:
                    logger.info("JPEG quality (30)")
                    if eps_v>0:
                        pic_all=[data_A,data_A_jpeg]
                        pic_t=['data_A','data_A_jpeg']
                        pic(pic_all,pic_t,1,3,title=model_name+'_'+A_name+'_noise_'+str(eps_v)+'_defense_'+filter) 
                    
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
                if mask>0:
                    data_A_mask = torch.tensor(data_A_mask) #tensor@cpu
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
                             data_A_mask,
                             data_A,
                             m0_1_m1_j,
                             m0_1_m1_0,
                             m0_0_m1_1,
                             m0_0_m1_j,
                             m0_j_m1_0,
                             m0_j_m1_1,
                             m0_j_m1_j]
                    pic_all_t=['data'         ,
                               'mask_rgb',
                               'data_A_mask',
                               'data_A',
                               'm0_1_m1_j',
                               'm0_1_m1_0',
                               'm0_0_m1_1',
                               'm0_0_m1_j',
                               'm0_j_m1_0',
                               'm0_j_m1_1',
                               'm0_j_m1_j']
                    if batch_idx==0 and eps_v>0:
                        pic(pic_all,pic_all_t,4,3,title=model_name+'_'+A_name+'_noise_'+str(eps_v)+'_defense_'+filter+'_'+mask_l[mask-1])
                    
                    
            case 'enhaceS':
                data_A_enhS = enhanceS(data_A[0,:,:,:],s=2*2.15)  #8./100.    #s is [0,1], expect output is on the range [0,1]
                data_A_enhS_norm=normalize(data_A_enhS)
                data_A_enhS_1 = enhanceS(data_A[0,:,:,:],s=255)      #s is [0,1], expect output is on the range [0,1]
                data_A_enhS_1_norm=normalize(data_A_enhS_1)
                # print('data_A_enhS max=',data_A_enhS.max(),' data_A_enhS min=',data_A_enhS.min())
                if batch_idx==0:
                    # print("EhanceS plot start")
                    if eps_v>0:
                        pic_all=[data_A,data_A_enhS_norm,data_A_enhS_1_norm]
                        pic_t=['data_A','data_A_enhS_norm','data_A_enhS_1_norm']
                        pic(pic_all,pic_t,1,3,title=model_name+'_'+A_name+'_noise_'+str(eps_v)+'_defense_'+filter) 
                    # print("EhanceS plot stop")
                    name_f_list.extend(['data_A_enhS_norm','data_A_enhS_1_norm'])
                for id,a in enumerate([data_A_enhS_norm,data_A_enhS_1_norm]):
                    output = model(a.to(device))
                    _, pred = output.max(1)
                    if batch_idx==0:
                        correct_f_list.append(pred.eq(target).sum())
                    else:
                        correct_f_list[id+correct_f_ini]+=pred.eq(target).sum()
                correct_f_ini=correct_f_ini+id+1                
               
                
            case 'combine': #enhance+jpeg+LL  fail no use
                data_A_enhS_1 = enhanceS(data_A[0,:,:,:],s=255)      #s is [0,255]#s is [0,1], expect output is on the range [0,1]
                data_A_enhS_1_norm=normalize(data_A_enhS_1)
                data_A_enhS_jpeg = (decode_jpeg(encode_jpeg((data_A_enhS_1_norm[0,:,:,:]*255).to(torch.uint8).cpu(),quality=30),device=device)/255.).to(torch.float).unsqueeze(0)
                data_A_enhS_jpeg_LL,_ = get_wavelet(data_A_enhS_jpeg)
                if batch_idx==0:
                    pic_all=[data_A,data_A_enhS_1_norm,data_A_enhS_jpeg,data_A_enhS_jpeg_LL]
                    pic_t=['data_A','data_A_enhS_1_norm','data_A_enhS_jpeg','data_A_enhS_jpeg_LL']
                    if eps_v>0:
                        pic(pic_all,pic_t,1,4,title=model_name+'_'+A_name+'_noise_'+str(eps_v)+'_defense_'+filter) 
                    name_f_list.extend(pic_t[1:])
                    
                    # print('correct_f_ini=',correct_f_ini)
                for id,a in enumerate(pic_all[1:]):#[data_A_enhS_1_norm,data_A_enhS_jpeg]):
                    output = model(a.to(device))
                    _, pred = output.max(1)
                    if batch_idx==0:
                        correct_f_list.append(pred.eq(target).sum()) 
                    else:
                        correct_f_list[id+correct_f_ini]+=pred.eq(target).sum() #前面ini=3個
                correct_f_ini=correct_f_ini+id+1
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
        if mask>0:
            ini=4
            for id,a in enumerate(pic_all[ini:]):
                # print('id =',id)
                a=a.to(device)
                output = model(a)
                _, pred = output.max(1)
                
                if batch_idx==0:
                    correct_hog_f_list.append(pred.eq(target).sum())
                    name_hog_f_list.append(pic_all_t[id+ini])
                else:    
                    correct_hog_f_list[id]+=pred.eq(target).sum()
            
                    

        # if batch_idx==2:
        # if batch_idx==round((len(data_loader)>>4)):
        #     break
        # break       
    
    acc = (100.*correct/total)
    
    strg=model_name+' Attack Finish data volume=({}/{})({:.2f}), eps= {} Acc=({:.2f}%) \n'.format(
            batch_idx,len(data_loader),1.0*batch_idx/len(data_loader),eps_v, acc) 
      
    for id,correct_ele in enumerate(correct_f_list): 
            strg=strg+'acc_'+name_f_list[id]+' = ({:.2f}%)\n'.format(100.*correct_ele/total)
    if mask>0:  
        savez_dict = dict()
        for id,correct_ele in enumerate(correct_hog_f_list): 
            strg=strg+'acc_'+name_hog_f_list[id]+' = ({:.2f}%) with '.format(100.*correct_ele/total)+mask_l[mask-1]+'\n'
        # for i in ['acc','correct_f_list','name_f_list','correct_hog_f_list','name_hog_f_list']:
        if torch.cuda.is_available():
            acc = acc.cpu().detach()
            correct_f_list = torch.stack(correct_f_list).cpu().detach()
            correct_hog_f_list = torch.stack(correct_hog_f_list).cpu().detach()
        acc = acc.numpy()
        correct_f_list=correct_f_list.numpy()
        correct_hog_f_list=correct_hog_f_list.numpy()
        savez_dict[ 'acc' ] = acc
        savez_dict[ 'correct_f_list' ] = correct_f_list
        savez_dict[ 'name_f_list' ] = name_f_list
        savez_dict[ 'correct_hog_f_list' ] = correct_hog_f_list
        savez_dict[ 'name_hog_f_list' ] = name_hog_f_list
            
        filename=model_name+'_'+A_name+'_noise_'+str(eps_v)+'_defense_'+filter+'_'+mask_l[mask-1]+'.npz'
        np.savez(model_name+'_'+A_name+'_noise_'+str(eps_v)+'_defense_'+filter+'_'+mask_l[mask-1]+'.npz', **savez_dict)
    logger.info(strg)
    
    # np.savez_compressed('example_savez_compressed.npz', my_arr4=arr4, my_arr5=arr5, my_arr6=arr6)
    
    
    # return acc,correct_f_list,name_f_list,correct_hog_f_list,name_hog_f_list
    # return filename

if __name__ == "__main__":
    gpuparser =  argparse.ArgumentParser()
    gpuparser.add_argument('-gpu',type=str,default='0',help='which gpus to use')
    gpuargs=gpuparser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuargs.gpu   #只看到第x張，或某幾張，這個時候這些會重新被index 0,1
    # Checking for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(int(gpuargs.gpu))
    print('gpu:',torch.cuda.current_device())
    args = parse_args()
    logger = get_logger(args['train']['log_path'], str(datetime.date.today())+ args['train']['log_file'])
    criterion = nn.CrossEntropyLoss()
    
    logger.info('Loading data start')
    train_dataloader,test_dataloader = get_dataset(batch_size=args['train']['batch_size'],
                             num_workers=args['train']['num_workers'])
    logger.info('Loading data finish')
    
    filterlist=['wavelet','jpeg']
    attacklist=['FGSM','iFGSM']
    for eps in np.linspace(0.015, 0.03, num=5, endpoint=True):
        for model_name in args['models']:
            model = get_model(model_name).to(device)
            optimizer,schedueler = get_optimizer(args['optimizer']['name'],
                                             model,
                                             args['optimizer']['lr'],
                                             args['optimizer']['weight_decay'])
            print_model(model)
            model, optimizer, schedueler, start_epoch= load_checkpoint(model,model_name,optimizer,schedueler,args['train']['checkpoint_path'],logger,device)


            model.eval() 
            gradients = None




            pathToFigure = 'Figure'

            if not os.path.isdir(pathToFigure):
                os.mkdir(pathToFigure)



                for at in attacklist[0]:
                    for fil in filterlist[1]:
                        for m in range(1):
                            attack(test_dataloader,at,eps_v=eps,filter=fil,mask=2+m)
        # acc = attack(test_dataloader,'FGSM',eps_v=0.015,filter='wavelet',hog=1)
        # acc = attack(test_dataloader,'FGSM',eps_v=0.015,filter='jpeg',hog=1)
        # break
        
        
        
    