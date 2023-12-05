#TODO
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F

import pywt
import copy
def get_dataset(batch_size, num_workers):
    train_loader = torch.utils.data.DataLoader(
        datasets.GTSRB(
            "./data",
            split="train",
            download=True,
            transform=transforms.Compose([
                        transforms.Resize([224, 224]),
                        transforms.ToTensor(),]),         
        ),
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.GTSRB(
            "./data",
            split="test",
            download=True,
            transform=transforms.Compose([
                        transforms.Resize([224, 224]),
                        transforms.ToTensor(),]),
        ),
        batch_size=1,
        shuffle=False, 
        num_workers=num_workers,
    )

    
                
                
    return train_loader,test_loader

 
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

#def get_wavelet(data_loader):
#     dataLL_loader = copy.deepcopy(data_loader)
#     dataHH_loader = copy.deepcopy(data_loader)
#     for idx in range(0,len(data_loader.dataset._samples)): 
#         LL, (LH, HL, HH) = pywt.dwt2(dataLL_loader.dataset[idx][0],'haar')#inputs(batch,3,224,224)
#         # print (torch.min(inputs[0,:,:,:]),torch.max(inputs[0,:,:,:]))
#         # print (LH.min(),LH.max(),len(LH))
#         # for i,a in enumerate([LL, LH, HL, HH]):  #3,114,114
#         #     ax = fig.add_subplot(1, 4, i + 1)
#             #  a=(a-a.min())/(a.max()-a.min())
#         #     ax.imshow(np.transpose(a,(1,2,0)), interpolation="nearest",vmin=0,vmax=1)
#         #     ax.set_title(titles[i], fontsize=10)
#         #     ax.set_xticks([])
#         #     ax.set_yticks([])
#         # fig.tight_layout()
#         # plt.show()
#         LL = (LL-LL.min())/(LL.max()-LL.min())      #range[0,1]
#         HH = (HH-HH.min())/(HH.max()-HH.min())      #range[0,1]
#         dataLL_loader.dataset[idx][0] = LL
#         dataHH_loader.dataset[idx][0] = HH
        


#     return dataLL_loader,dataHH_loader