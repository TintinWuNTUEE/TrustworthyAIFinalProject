import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
pathToFigure = '../Figure'

if not os.path.isdir(pathToFigure):
    os.mkdir(pathToFigure)

def pic(pic_l,pic_l_t,row,col,title,filename=None,cmap=0):
    fig=plt.figure()
    for id,a in enumerate(pic_l):  
        # print('subplot ',pic_l_t[id])
        ax = fig.add_subplot(row, col, id+1)
        if torch.is_tensor(a):
            if a.device == 'cpu':
                a=a.detach().numpy()
            else:
                a=a.cpu().detach().numpy()
        if a.ndim==4:
            a=a[0,:,:,:]
        if a.shape[2]==3:
            ax.imshow(a,vmin=0,vmax=1)
        else:
          ax.imshow(np.transpose(a,(1,2,0)),vmin=0,vmax=1)
        ax.set_title(pic_l_t[id], fontsize=10)
        # print(pic_all_t[id],type(a))
            #   print(pic_all_t[id],a.shape)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle(title)
    if filename==None:
        filename=title+'.jpg'
    fig.tight_layout()
    # plt.show()
    save_path = os.path.join(pathToFigure, filename)
    plt.savefig(save_path)

def pil2cv2_grayscale(img):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    return img

def normalize(img):
    img=(img-img.min())/(img.max()-img.min())
    return img