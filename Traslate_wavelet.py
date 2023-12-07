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
def normalize(img):
    img=(img-img.min())/(img.max()-img.min())
    return img
def attack(data_loader,A_name,eps_v=0.015,filter=0,hog=0):

    
    total = 0
    correct = 0
    correct_A =0
    correct_A_LL = 0
    correct_A_HH = 0
    acc_A_LL = 200
    acc_A_HH = 200
    logger.info('Attack start'+model_name+ ' '+A_name)
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
        ini=id+1
        
        
        if filter==1:
            
            data_A_LL,data_A_HH = get_wavelet(data_A)
            data_A_LL,data_A_HH=normalize(data_A_LL.to(device)),normalize(data_A_HH.to(device))    #沒用到mask
            # data_A_grb=data_A[:, [1, 2, 0], :, :]
            # # print(data_A_grb.size())
            # data_A_grb_LL,data_A_grb_HH = get_wavelet(data_A_grb)
            # data_A_grb_LL,data_A_grb_HH = normalize(data_A_grb_LL.to(device)),normalize(data_A_grb_HH.to(device))
            
            # blurer= v2.GaussianBlur(kernel_size=9, sigma=( 5.))
            # data_A_blur = blurer(data_A)
            # data_A_blur_LL,data_A_blur_HH = get_wavelet(data_A_blur)
            # data_A_blur_LL,data_A_blur_HH = normalize(data_A_blur_LL.to(device)),normalize(data_A_blur_HH.to(device))
            data_A_jpeg =(decode_jpeg(encode_jpeg((data_A[0,:,:,:]*255).to(torch.uint8).cpu(),quality=30),device=device)/255.).to(torch.float).unsqueeze(0)
            
            # print('data_A_jpeg type:',type(data_A_jpeg),'\n shape',data_A_jpeg.size())
            if hog ==0:
                if batch_idx==0:
                    logger.info("JPEG quality (30)")
                    # name_f_list.extend(['data_A_grb','data_A_grb_LL','data_A_grb_HH',
                    #                 'data_A_blur','data_A_blur_LL','data_A_blur_HH','data_A_LL','data_A_HH'])
                    name_f_list.extend(['data_A_jpeg','data_A_LL','data_A_HH'])
                    fig=plt.figure()
                for id,a in enumerate([data_A_jpeg,data_A_LL,data_A_HH]):
                    # [data_A_grb,data_A_grb_LL,data_A_grb_HH,
                    #                    data_A_blur,data_A_blur_LL,data_A_blur_HH,
                    #                    data_A_LL,data_A_HH]) 
                    output = model(a.to(device))
                    _, pred = output.max(1)
                    if batch_idx==0:
                        correct_f_list.append(pred.eq(target).sum())
                        ax = fig.add_subplot(3, 3, id+ 1)
                        a = a[0,:,:,:].cpu().detach().numpy()
                        ax.imshow(np.transpose(a,(1,2,0)), interpolation="nearest",vmin=0,vmax=1)
                        ax.set_title(name_f_list[id+ini], fontsize=10)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        fig.tight_layout()
                        # plt.show()
                        save_path = os.path.join(pathToFigure, f'blur.jpg')
                        plt.savefig(save_path)
                    else:
                        correct_f_list[id+ini]+=pred.eq(target).sum()

                # output_A_LL = model(data_A_LL.to(device) )
                # output_A_HH = model(data_A_HH.to(device) ) 
                # _, pred_A_LL = output_A_LL.max(1)
                # _, pred_A_HH = output_A_HH.max(1)


                # correct_A_LL += pred_A_LL.eq(target).sum()#.item()
                # correct_A_HH  += pred_A_HH.eq(target).sum()#.item()
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
        if batch_idx==round((len(data_loader)>>4)):
            break
           
            
                

    acc = 100.*correct/total
    # acc_A = 100.*correct_A/total  
    strg=model_name+'Attack Finish data volume=({}/{})({:.2f}), eps= {} Acc=({:.2f}%) \n'.format(
            batch_idx,len(data_loader),1.0*batch_idx/len(data_loader),eps_v, acc) 
      
    for id,correct_ele in enumerate(correct_f_list): 
            strg=strg+'acc_'+name_f_list[id]+' = ({:.2f}%)\n'.format(100.*correct_ele/total)
    if filter ==1 and hog==1:  
        # if hog ==0:
            # acc_A_LL = 100.*correct_A_LL/total
            # acc_A_HH = 100.*correct_A_HH/total
        
            # strg=strg+'acc_A_LL =({:.2f}%), acc_A_HH =({:.2f}%)'.format(acc_A_LL,acc_A_HH)
            
        # else:
            #  strg=strg+model_name+' Attack Finish, eps= = {} Acc=({:.2f}%),'.format(eps_v, acc)
            for id,correct_ele in enumerate(correct_hog_f_list): 
                strg=strg+'acc_'+name_hog_f_list[id]+' = ({:.2f}%)\n'.format(100.*correct_ele/total)
            
    logger.info(strg)
    
        
    return acc

if __name__ == '__main__':   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    logger = get_logger('./logs', str(datetime.date.today())+'train.log')
    
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
    

    # loss_test,acc_test = test(test_loader)                      #calculate acc
    

    # plt.figure(figsize = (4,5))
    
   
    
    
    # 
    for eps in np.linspace(0.015, 0.1, num=5, endpoint=True):
        acc = attack(test_loader,'FGSM',eps_v=eps,filter=1)
    acc = attack(test_loader,'FGSM',eps_v=0,filter=1)
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
    