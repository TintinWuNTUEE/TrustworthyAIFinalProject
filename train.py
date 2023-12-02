import yaml
import torch
import torch.nn as nn
from models import get_model
from dataset import get_dataset,get_wavelet,FGSM
from optimizer import get_optimizer
from checkpoint import save_checkpoint,load_checkpoint
from logger import get_logger

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
    model = get_model(model_name)
    
    # optimizer,schedueler = get_optimizer(args['optimizer']['name'],
    #                                      model,
    #                                      args['optimizer']['lr'],
    #                                      args['optimizer']['weight_decay'])
    #criterion = nn.CrossEntropyLoss()
    epochs = args['train']['epochs']
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
def attack(data_loader,A_name):
    # for p in model.parameters():
    #     p.requires_grad = True
    # if model_name =='resnet18':
    #     gradients = None
    #     model.conv1.register_full_backward_hook(backward_hook, prepend=False)
    
    total = 0
    correct = 0
    correct_A =0
    correct_A_LL = 0
    correct_A_HH = 0
    logger.info('Attack start')
    for batch_idx, (data, target) in enumerate(data_loader):

        data, target =data.to(device), target.to(device) 
        
        data.requires_grad = True
        output= model(data)
        loss =  criterion(output, target)
        #print("OutputSize",output.size())
        #print("loss=",loss)
        #print("Output",output)
        model.zero_grad() #zero all existing gradients
        #optimizer.zero_grad() #zero all existing gradients
        loss.backward(retain_graph=True)
        _, pred = output.max(1) #normal data classified result
        if batch_idx <10:
            print("target=",target)
            print("predict = ",pred)
        total += target.size(0)
        if(A_name=='FGSM'):
            eps_v = 0.015
            data_A = FGSM (data ,eps_v,data.grad.data)
            # data_A = FGSM (data ,eps_v,gradients[0])

        

        data_A_LL,data_A_HH = get_wavelet(data_A)
        output_A = model(data_A)
        output_A_LL = model(data_A_LL)
        output_A_HH = model(data_A_HH) 
        _, pred_A = output_A.max(1)
        _, pred_A_LL = output_A_LL.max(1)
        _, pred_A_HH = output_A_HH.max(1)
        #print("pred.eq type,",type(pred.eq(target).sum()))
        correct += pred.eq(target).sum()#.item()
        correct_A += pred_A.eq(target).sum()#.item()
        correct_A_LL += pred_A_LL.eq(target).sum()#.item()
        correct_A_HH  += pred_A_HH.eq(target).sum()#.item()
    acc = 100.*correct/total
    acc_A = 100.*correct_A/total
    acc_A_LL = 100.*correct_A_LL/total
    acc_A_HH = 100.*correct_A_HH/total
    logger.info('Attack Finish, Acc=({:.2f}%), acc_A =({:.2f}%), acc_A_LL =({:.2f}%), acc_A_HH =({:.2f}%)'.format(
         100. * acc, 100. *acc_A,100. *acc_A_LL,100. *acc_A_HH))
    print('eps=',eps_v,' acc =',acc,', acc_A =',acc_A,', acc_A_LL=',acc_A_LL,', acc_A_HH=',acc_A_HH)
    return acc,acc_A,acc_A_LL,acc_A_HH

# def backward_hook(module, grad_input, grad_output): 
#     global gradients # refers to the variable in the global scope 

 
#     gradients=grad_input
#     #print("grad_input",grad_input)

#     return grad_input        

if __name__ == "__main__":
    
    args = parse_args()
    criterion = nn.CrossEntropyLoss()
    
    train_dataloader,test_dataloader = get_dataset(batch_size=args['train']['batch_size'],
                             num_workers=args['train']['num_workers'])
    
    logger = get_logger(args['train']['log_path'], args['train']['log_file'])
    
    #如果已經有training檔案就直接load
    # for model_name in args['models']:
    #     if not os.path.isfile([args['train']['checkpoint_path'],'/',model_name]):
    #         input()
    #         train()
    #     else:
    #         model = get_model(model_name])
    #         model.load_state_dict(torch.load(model_name))          #一般正常load model方式
    #         # checkpoint = torch.load([args['train']['checkpoint_path'],'/',model_name])
    #          #model.load_state_dict(checkpoint['model_state_dict'])
    #          #model, optimizer, schedueler, start_epoch=load_checkpoint(model,model_name,optimizer,schedueler,args['train']['checkpoint_path'],logger)
    for model_name in args['models']:
        model = get_model(model_name).to(device)
        optimizer,schedueler = get_optimizer(args['optimizer']['name'],
                                         model,
                                         args['optimizer']['lr'],
                                         args['optimizer']['weight_decay'])
        # model_path=args['train']['checkpoint_path']+'/'+model_name+'/epoch_112.pth'
        # print(model_path)
        # model.load_state_dict(torch.load(model_path,map_location = torch.device('cpu'))["model"])         #一般正常load model方式
        model, optimizer, schedueler, start_epoch= load_checkpoint(model,model_name,optimizer,schedueler,args['train']['checkpoint_path'],logger)
        
        #print("start_epoch",start_epoch)
        model.eval() 
        gradients = None
        
        print_model(model)
        
        
        
    
    acc,acc_A,acc_A_LL,acc_A_HH = attack(test_dataloader,'FGSM')
    