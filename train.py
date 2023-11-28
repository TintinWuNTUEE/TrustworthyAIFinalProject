import yaml
import torch
import torch.nn as nn
from models import get_model
from dataset import get_dataset,get_wavelet,FGSM
from optimizer import get_optimizer
from checkpoint import save_checkpoint,load_checkpoint
from logger import get_logger

import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def parse_args(file_path='./config.yaml'):
    with open(file_path, 'r') as file:
        args = yaml.safe_load(file)
    return args
def train():   
    #args = parse_args()
    logger = get_logger(args['train']['log_path'], args['train']['log_file'])
    
    for model_name in args['models']:
        train_single_model(args,model_name,train_dataloader,test_dataloader,logger)
    return
def train_single_model(args,model_name,train_dataloader,test_dataloader,logger):
    model = get_model(model_name)
    
    optimizer,schedueler = get_optimizer(args['optimizer']['name'],
                                         model,
                                         args['optimizer']['lr'],
                                         args['optimizer']['weight_decay'])
    #criterion = nn.CrossEntropyLoss()
    epochs = args['train']['epochs']
    model, optimizer, schedueler, start_epoch= load_checkpoint(model,model_name,optimizer,schedueler,args['train']['checkpoint_path'],logger)
    acc = 0.0
    best_acc = 0.0
    for epoch in range(start_epoch,epochs):
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            schedueler.step()
            if batch_idx % args['train']['log_interval'] == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_dataloader.dataset),
                        100. * batch_idx / len(train_dataloader), loss.item()))
        if batch_idx % args['train']['val_interval'] == 0:
            acc=validataion(args,model,test_dataloader,criterion,epoch,logger)
            if acc > best_acc:
                best_acc = acc
                save_checkpoint(args['train']['checkpoint_path'],model_name,model,optimizer,epoch,schedueler)
        if batch_idx % args['train']['save_interval'] == 0:
                save_checkpoint(args['train']['checkpoint_path'],model_name,model,optimizer,epoch,schedueler)
    return
def validataion(args,model,test_loader,criterion,epoch,logger):
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        acc = pred.eq(target.view_as(pred)).float().mean()/len(test_loader)
        logger.info('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
                epoch, batch_idx * len(data), len(test_loader.dataset),
                100. * batch_idx / len(test_loader), loss.item(), acc.item()))
    return acc
def attack(test_loader,A_name):
    total = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data.required_grad = True
        output= model(data)
        loss =  criterion(output, target)
        model.zero_grad() #zero all existing gradients
        loss.backward()
        _, pred = output.max(1) #normal data classified result

        total += target.size(0)
        if(A_name=='FGSM'):
            eps_v = 0.015
            data_A = FGSM (data ,eps_v,data.grad.data)
        

        data_A_LL,data_A_HH = get_wavelet(data_A)
        output_A = model(data_A)
        output_A_LL = model(data_A_LL)
        output_A_HH = model(data_A_HH) 
        _, pred_A = output_A.max(1)
        _, pred_A_LL = output_A_LL.max(1)
        _, pred_A_HH = output_A_HH.max(1)
        correct +=pred.eq(target).sum.item()
        correct_A += pred_A.eq(target).sum().item()
        correct_A_LL += pred_A_LL.eq(target).sum().item()
        correct_A_HH  += pred_A_HH.eq(target).sum().item()
    acc = 100.*correct/total
    acc_A = 100.*correct_A/total
    acc_A_LL = 100.*correct_A_LL/total
    acc_A_HH = 100.*correct_A_HH/total
    print('eps=',eps_v,' acc =',acc,', acc_A =',acc_A,', acc_A_LL=',acc_A_LL,', acc_A_HH=',acc_A_HH)
        

if __name__ == "__main__":
    
    args = parse_args()
    criterion = nn.CrossEntropyLoss()

    for model_name in args['models']:
        if not os.path.isfile(model_name):
            train()
        else:
             model = get_model(model_name)
             model.load_state_dict(torch.load(model_name))          #一般正常load model方式
    
    train_dataloader,test_dataloader = get_dataset(batch_size=args['train']['batch_size'],
                             num_workers=args['train']['num_workers'])
    
    #trainLL_loader,trainHH_loader = get_wavelet(train_dataloader)
    testLL_loader,testHH_loader = get_wavelet(test_dataloader)