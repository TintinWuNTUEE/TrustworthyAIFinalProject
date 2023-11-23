import yaml
import torch
import torch.nn as nn
from models import get_model
from dataset import get_dataset
from optimizer import get_optimizer
from checkpoint import save_checkpoint,load_checkpoint
from logger import get_logger
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def parse_args(file_path='./config.yaml'):
    with open(file_path, 'r') as file:
        args = yaml.safe_load(file)
    return args
def train():   
    args = parse_args()
    logger = get_logger(args['train']['log_path'], args['train']['log_file'])
    train_dataloader,test_dataloader = get_dataset(batch_size=args['train']['batch_size'],
                             num_workers=args['train']['num_workers'])
    for model_name in args['models']:
        train_single_model(args,model_name,train_dataloader,test_dataloader,logger)
    return
def train_single_model(args,model_name,train_dataloader,test_dataloader,logger):
    model = get_model(model_name)
    
    optimizer,schedueler = get_optimizer(args['optimizer']['name'],
                                         model,
                                         args['optimizer']['lr'],
                                         args['optimizer']['weight_decay'])
    criterion = nn.CrossEntropyLoss()
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

if __name__ == "__main__":
    train()