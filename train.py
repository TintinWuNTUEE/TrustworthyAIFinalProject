import yaml
import torch
import torch.nn as nn
from models import get_model
from dataset import get_dataset,get_wavelet,FGSM
from optimizer import get_optimizer
from checkpoint import save_checkpoint,load_checkpoint
from logger import get_logger
from utils.hog import mask_hog

import os

import torchvision.models as models
import matplotlib.pyplot as plt
import random
import numpy as np
from torchsummary import summary
from thop import profile

# Checking for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
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
def normalize(img):
    img=(img-img.min())/(img.max()-img.min())
    return img
# def attack(data_loader,A_name):
#     # for p in model.parameters():
#     #     p.requires_grad = True
#     # if model_name =='resnet18':
#     #     gradients = None
#     #     model.conv1.register_full_backward_hook(backward_hook, prepend=False)
    
#     total = 0
#     correct = 0
#     correct_A =0
#     correct_A_LL = 0
#     correct_A_HH = 0
#     acc_A_LL = 200
#     acc_A_HH = 200
#     logger.info('Attack start')
#     filter = 1
    
#     for batch_idx, (data, target) in enumerate(data_loader):

#         data, target =data.to(device), target.to(device) 
        
#         data.requires_grad = True
#         output= model(data)
#         loss =  criterion(output, target)
#         #print("OutputSize",output.size())
#         #print("loss=",loss)
#         #print("Output",output)
#         model.zero_grad() #zero all existing gradients
#         #optimizer.zero_grad() #zero all existing gradients
#         loss.backward(retain_graph=True)
#         _, pred = output.max(1) #normal data classified result
#         if batch_idx <10:
#             print("target=",target)
#             print("predict = ",pred)
#         total += target.size(0)
#         if(A_name=='FGSM'):
#             eps_v = 0.015
#             data_A = FGSM (data ,eps_v,data.grad.data)
#             # data_A = FGSM (data ,eps_v,gradients[0])

#         output_A = model(data_A)
#         _, pred_A = output_A.max(1)
#         correct += pred.eq(target).sum()#.item()
#         correct_A += pred_A.eq(target).sum()#.item()
#         acc = 100.*correct/total
#         acc_A = 100.*correct_A/total
        
#         if filter==1:
#             data_A_LL,data_A_HH = get_wavelet(data_A)
#             output_A_LL = model(data_A_LL.to(device) )
#             output_A_HH = model(data_A_HH.to(device) ) 
#             _, pred_A_LL = output_A_LL.max(1)
#             _, pred_A_HH = output_A_HH.max(1)
        
        
#             correct_A_LL += pred_A_LL.eq(target).sum()#.item()
#             correct_A_HH  += pred_A_HH.eq(target).sum()#.item()
    
#     if filter==1:
#         acc_A_LL = 100.*correct_A_LL/total
#         acc_A_HH = 100.*correct_A_HH/total
#         logger.info('Attack Finish, eps= {} Acc=({:.2f}%), acc_A =({:.2f}%), acc_A_LL =({:.2f}%), acc_A_HH =({:.2f}%)'.format(
#         eps_v, acc, acc_A,acc_A_LL,acc_A_HH))
#     else:
#         logger.info('Attack Finish, eps= {} Acc=({:.2f}%), acc_A =({:.2f}%)'.format(
#         eps_v, acc, acc_A))
#     return acc,acc_A,acc_A_LL,acc_A_HH
    
def attack(data_loader,A_name,eps_v=0.015,filter=0,hog=0):

    
    total = 0
    correct = 0
    correct_A =0
    correct_A_LL = 0
    correct_A_HH = 0
    acc_A_LL = 200
    acc_A_HH = 200
    logger.info('Attack start')
    # filter = 1
    # hog = 0
    name_hog_f_list=[]
    correct_hog_f_list =[]
    name_f_list=[]
    correct_f_list=[]
    # correct_class=[[] for i in range(43)]
    for batch_idx, (data, target) in enumerate(data_loader):

        data, target =data.to(device), target.to(device) 
        
        data.requires_grad = True
        # print ("data range:",torch.min(data[0,:,:,:]),"-",torch.max(data[0,:,:,:]))
        output= model(data)
        loss =  criterion(output, target)
        #print("OutputSize",output.size())
        #print("loss=",loss)
        #print("Output",output)
        model.zero_grad() #zero all existing gradients
        #optimizer.zero_grad() #zero all existing gradients
        loss.backward(retain_graph=True)
        _, pred = output.max(1) #normal data classified result
        correct += pred.eq(target).sum()#.item()
        # correct_class[target][0]+= pred.eq(target).sum()
        # if batch_idx <10:
        #     print("target=",target)
        #     print("predict = ",pred)
        total += target.size(0)
        
        if(A_name=='FGSM'):
            # eps_v = 0.015
            
            data_A,noise = FGSM (data ,eps_v,data.grad.data)
            noise_LL,noise_HH = get_wavelet(noise)
            data_A_noise_LL =normalize( data + noise_LL.to(device))
            data_A_noise_HH =normalize( data + noise_HH.to(device))

        
        # ini=0
        # output_A = model(data_A)       #沒用到mask
        # _, pred_A = output_A.max(1)
        # correct_A += pred_A.eq(target).sum()#.item()
        if batch_idx==0:
            name_f_list=['data_A','data_A_noise_LL','data_A_noise_HH']
        for id,a in enumerate([data_A,data_A_noise_LL,data_A_noise_HH]):  
            output = model(a.to(device))
            _, pred = output.max(1)
            if batch_idx==0:
                correct_f_list.append(pred.eq(target).sum())
                
            else:
                correct_f_list[id]+=pred.eq(target).sum()
        
        
        
        if filter==1:
            data_A_LL,data_A_HH = get_wavelet(data_A)
            data_A_LL,data_A_HH=normalize(data_A_LL),normalize(data_A_HH)    #沒用到mask
            if hog ==0:
                output_A_LL = model(data_A_LL.to(device) )
                output_A_HH = model(data_A_HH.to(device) ) 
                _, pred_A_LL = output_A_LL.max(1)
                _, pred_A_HH = output_A_HH.max(1)


                correct_A_LL += pred_A_LL.eq(target).sum()#.item()
                correct_A_HH  += pred_A_HH.eq(target).sum()#.item()
        if hog ==1 :
            ini = 0
            # data_hog = mask_hog(data_A)
            hog_mask,hog_mask_rgb,img_hog_mask_rgb= mask_hog(data_A[0,:,:,:])
            
            
            pic_all_ori = [data]
            pic_all_ori_t = ['data']
            if batch_idx==0:
                fig=plt.figure()
                for id,a in enumerate(pic_all_ori):  #3,114,114
                    
                    ax = fig.add_subplot(5, 3, id+ 1)
                    a = a[0,:,:,:].cpu().detach().numpy()
                    ax.imshow(np.transpose(a,(1,2,0)), interpolation="nearest",vmin=0,vmax=1)
                    ax.set_title(pic_all_ori_t[id], fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
            # hog_mask=normalize(hog_mask)
            # hog_mask[hog_mask>0]=1
            img_hog_mask_rgb = normalize(img_hog_mask_rgb) 
            img_hog_mask_rgb = np.transpose(img_hog_mask_rgb,(2,0,1))
            hog_mask_rgb =normalize(hog_mask_rgb) #224,224,3
            hog_mask_rgb = np.transpose(hog_mask_rgb,(2,0,1))#3,224,224
            # hog_mask_rgb_m=hog_mask_rgb.copy()
            # hog_mask_rgb_m[hog_mask_rgb_m>0]=1
            hog_mask_bar= 1-hog_mask_rgb
            
            
                
                
            # hog_mask = np.array([hog_mask]*3)
            pic_all=[ hog_mask_rgb,img_hog_mask_rgb]#, hog_mask_rgb, img_hog_mask_rgb]
            pic_all_t=[ 'hog_mask_rgb','img_hog_mask_rgb']#, 'hog_mask_rgb', 'img_hog_mask_rgb']
            for id,a in enumerate(pic_all):  
                output = model(torch.tensor(a).unsqueeze(0).to(device))
                _, pred = output.max(1)
                if batch_idx==0:
                    correct_hog_f_list.append(pred.eq(target).sum())
                    name_hog_f_list.append(pic_all_t[id])
                else:
                    correct_hog_f_list[id]+=pred.eq(target).sum()
                ax = fig.add_subplot(5, 3, (id) + 2)
            #   print(pic_all_t[id]," range: ",a.max(),"-",a.min())
                
              # ax.imshow(np.transpose(a,(1,2,0)), interpolation="nearest",vmin=0,vmax=1)
                if a.shape[2]==3:
                    ax.imshow(a,vmin=0,vmax=1)
                else:
                  ax.imshow(np.transpose(a,(1,2,0)),vmin=0,vmax=1)
                ax.set_title(pic_all_t[id], fontsize=10)
                # print(pic_all_t[id],type(a))
            #     print(pic_all_t[id],a.shape)
                ax.set_xticks([])
                ax.set_yticks([])
            ini = id+1
                
            
            
            # ax = fig.add_subplot(len(pic_all)+2, 2, 2)
            # ax.hist(a.flatten(), linewidth=0.5, edgecolor="white")
            # ax.set_title(pic_all_t[id]+" hist", fontsize=10)
            data_A_mask = hog_mask_rgb*data_A[0,:,:,:].cpu().detach().numpy()        #沒用到filter
            data_A_bar_mask = hog_mask_bar*data_A[0,:,:,:].cpu().detach().numpy()
            # print('data_A_mask shape = ',data_A_mask.shape)
            # print('data_A_bar_mask shape = ',data_A_bar_mask.shape)
            data_A_m1_LL,data_A_m1_HH =get_wavelet(data_A_mask)
            data_A_m0_LL,data_A_m0_HH =get_wavelet(data_A_bar_mask)
            
            m0_1_m1_LL = normalize(torch.tensor(data_A_bar_mask).unsqueeze(0)+data_A_m1_LL)
            m0_1_m1_HH = normalize(torch.tensor(data_A_bar_mask).unsqueeze(0)+data_A_m1_HH)

            m0_0_m1_1 = normalize(torch.tensor(data_A_mask) .unsqueeze(0))   #m0_0_m1_1
            m0_0_m1_LL = normalize(data_A_m1_LL)  #m0_0_m1_LL
            m0_0_m1_HH = normalize(data_A_m1_HH)  #m0_0_m1_HH

            m0_LL_m1_1 = normalize(data_A_m0_LL+torch.tensor(data_A_mask).unsqueeze(0))
            m0_LL_m1_LL = data_A_LL
            m0_LL_m1_HH= normalize(data_A_m0_LL+ data_A_m1_HH)
            m0_HH_m1_1 = normalize(data_A_m0_HH+torch.tensor(data_A_mask).unsqueeze(0))
            m0_HH_m1_LL= normalize(data_A_m0_HH + data_A_m1_LL)
            m0_HH_m1_HH = data_A_HH
            
            pic_all=[data_A,
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
            pic_all_t=[ 'data_A',
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
            
            
            for id,a in enumerate(pic_all):
                if pic_all_t[id]!='data_A':
                    a=a.to(device)
                    # print(pic_all_t[id])
                    output = model(a)
                _, pred = output.max(1)
                if batch_idx==0:
                    if pic_all_t[id]!='data_A':
                        correct_hog_f_list.append(pred.eq(target).sum())
                        name_hog_f_list.append(pic_all_t[id])
                    ax = fig.add_subplot(5, 3, (id) + 4)
                    if torch.is_tensor(a):
                        if a.device == 'cpu':
                            a=a.detach().numpy()  
                        else:
                            a=a.cpu().detach().numpy()
                        a=a[0,:,:,:]              #3,114,114
                    # print(pic_all_t[id]," range: ",a.max(),"-",a.min())
                    # ax.imshow(np.transpose(a,(1,2,0)), interpolation="nearest",vmin=0,vmax=1)
                    ax.imshow(np.transpose(a,(1,2,0)),vmin=0,vmax=1)
                    ax.set_title(pic_all_t[id], fontsize=10)
                    #print('ini',ini)
                    # print(pic_all_t[id],type(a))
                    # print(pic_all_t[id],a.shape)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    if pic_all_t[id]!='data_A':
                        correct_hog_f_list[id+ini-1]+=pred.eq(target).sum()
                    
            if batch_idx==0:
                fig.tight_layout()
                # plt.show()
                save_path = os.path.join(pathToFigure, f'hogData.jpg')
                plt.savefig(save_path)
        
         
            
                

    acc = 100.*correct/total
    # acc_A = 100.*correct_A/total  
    strg=model_name+'Attack Finish, eps= {} Acc=({:.2f}%) \n'.format(
            eps_v, acc) 
      
    for id,correct_ele in enumerate(correct_f_list): 
            strg=strg+'acc_'+name_f_list[id]+' = ({:.2f}%)\n'.format(100.*correct_ele/total)
    if filter ==1:  
        if hog ==0:
            acc_A_LL = 100.*correct_A_LL/total
            acc_A_HH = 100.*correct_A_HH/total
        
            strg=strg+'acc_A_LL =({:.2f}%), acc_A_HH =({:.2f}%)\n'.format(acc_A_LL,acc_A_HH)
            
        else:
            # strg=strg+model_name+' Attack Finish, eps= = {} Acc=({:.2f}%),'.format(eps_v, acc)
            for id,correct_ele in enumerate(correct_hog_f_list): 
                strg=strg+'acc_'+name_hog_f_list[id]+' = ({:.2f}%)\n'.format(100.*correct_ele/total)
            
    logger.info(strg)
    
        
    return acc

# def backward_hook(module, grad_input, grad_output): 
#     global gradients # refers to the variable in the global scope 

 
#     gradients=grad_input
#     #print("grad_input",grad_input)

#     return grad_input        

if __name__ == "__main__":
    
    args = parse_args()
    logger = get_logger(args['train']['log_path'], args['train']['log_file'])
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
    
        # acc,acc_A,acc_A_LL,acc_A_HH = attack(test_dataloader,'FGSM')
        for eps in np.linspace(0.015, 0.1, num=5, endpoint=True):
            acc = attack(test_dataloader,'FGSM',filter=1,hog=1,eps_v=eps)
    