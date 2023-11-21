import yaml
import torch.nn as nn
from models import get_model
from dataset import get_dataset
from optimizer import get_optimizer
def parse_args(file_path='./config.yaml'):
    with open(file_path, 'r') as file:
        args = yaml.safe_load(file)
    return args
def train():   
    args = parse_args()
    train_dataloader,test_dataloader = get_dataset(batch_size=args['train']['batch_size'],
                             num_workers=args['train']['num_workers'])
    for model_name in args['model']:
        train_single_model(args,model_name,train_dataloader,test_dataloader)
    return
def train_single_model(args,model_name,train_dataloader,test_dataloader):
    model = get_model(model_name)
    optimizer,schedueler = get_optimizer(args['train']['optimizer'],
                                         args['train']['lr'],
                                         args['train']['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    epochs = args['train']['epoch']
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            schedueler.step()
            if batch_idx % args['train']['log_interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_dataloader.dataset),
                        100. * batch_idx / len(train_dataloader), loss.item()))
            if batch_idx % args['train']['val_interval'] == 0:
                validataion(args,model,test_dataloader,criterion,epoch)
    return
def validataion(args,model,test_loader,criterion,epoch):
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)
        loss = criterion(output, target)
        if batch_idx % args['train']['log_interval'] == 0:
            print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), loss.item()))
    return

if __name__ == "__main__":
    train()