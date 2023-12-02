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
import os
import copy
import pywt
#from utils import progress_bar

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

        if torch.cuda.is_available():
            LL, (LH, HL, HH) = pywt.dwt2(image.cpu().detach().numpy(),'haar')#inputs(batch,3,224,224)
        else:
            LL, (LH, HL, HH) = pywt.dwt2(image.detach().numpy(),'haar')#inputs(batch,3,224,224)
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
        LL = (LL-LL.min())/(LL.max()-LL.min())      #range[0,1]
        HH = (HH-HH.min())/(HH.max()-HH.min())      #range[0,1]
        LL = F.interpolate(torch.tensor(LL),mode='area',size=[224,224])
        HH = F.interpolate(torch.tensor(HH),mode='area',size=[224,224])
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
    return attack_i    
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
    acc_A_LL = 200
    acc_A_HH = 200
    logger.info('Attack start')
    filter = 1
    
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

        output_A = model(data_A)
        _, pred_A = output_A.max(1)
        correct += pred.eq(target).sum()#.item()
        correct_A += pred_A.eq(target).sum()#.item()
        acc = 100.*correct/total
        acc_A = 100.*correct_A/total
        
        if filter==1:
            data_A_LL,data_A_HH = get_wavelet(data_A)
            output_A_LL = model(data_A_LL.to(device) )
            output_A_HH = model(data_A_HH.to(device) ) 
            _, pred_A_LL = output_A_LL.max(1)
            _, pred_A_HH = output_A_HH.max(1)
        
        
            correct_A_LL += pred_A_LL.eq(target).sum()#.item()
            correct_A_HH  += pred_A_HH.eq(target).sum()#.item()
    
    if filter==1:
        acc_A_LL = 100.*correct_A_LL/total
        acc_A_HH = 100.*correct_A_HH/total
        logger.info('Attack Finish, eps= {} Acc=({:.2f}%), acc_A =({:.2f}%), acc_A_LL =({:.2f}%), acc_A_HH =({:.2f}%)'.format(
        eps_v, acc, acc_A,acc_A_LL,acc_A_HH))
    else:
        logger.info('Attack Finish, eps= {} Acc=({:.2f}%), acc_A =({:.2f}%)'.format(
        eps_v, acc, acc_A))
    return acc,acc_A,acc_A_LL,acc_A_HH

if __name__ == '__main__':   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    logger = get_logger('./logs', 'train.log')
    
    model_name='GTSRB_simpleCNN.pt'
    model = Net().to(device)
    # model.load_state_dict(torch.load("mnist_model.pth", map_location="cpu"))
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
            download=True,
            transform=transforms.Compose([
                        transforms.Resize([224, 224]),
                        transforms.ToTensor()]),
        ),
        batch_size=1,
        shuffle=True,
    )

    pathToFigure = 'Figure'
    # pathToModel = 'Model'
    if not os.path.isdir(pathToFigure):
        os.mkdir(pathToFigure)
    # if not os.path.isdir(pathToModel):
        # os.mkdir(pathToModel)
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
        # checkpoint = torch.load(model_name)
        # model.load_state_dict(checkpoint['net'])                #load客製化的model
        model.load_state_dict(torch.load(model_name))          #一般正常load model方式
    

    loss_test,acc_test = test(test_loader)                      #calculate acc
    

    plt.figure(figsize = (4,5))
    
   
    
    pathToFigure = 'Figure'
    if not os.path.isdir(pathToFigure):
        os.mkdir(pathToFigure)

    fig=plt.figure(figsize = (5,10))
    acc,acc_A,acc_A_LL,acc_A_HH = attack(test_loader,'FGSM')
    
    """
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
    """
    