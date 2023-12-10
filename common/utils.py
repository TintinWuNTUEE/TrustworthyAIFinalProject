import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from rembg import remove
import time
import PIL.Image
# from dataset import get_dataset
pathToFigure ='../Figure'

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
    denom=(img.max()-img.min())
    if denom>0:
        img=(img-img.min())/denom
    return img

def enhanceS(img,s):
    # print(img.size())
    if torch.cuda.is_available():
        img=np.transpose(img.cpu().detach().numpy(),(1,2,0))
    
        img=np.transpose(img.detach().numpy(),(1,2,0))
    HLS = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HLS)
    # HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # print('HLS type',HLS.dtype,' max=',np.max(np.max(HLS,axis=0),axis=0),' min= ',np.min(np.min(HLS,axis=0),axis=0))
    
    HLS[:,:,2]=np.clip(HLS[:,:,2]+s,0,255)    #0-360,0-1,0-1 #0-180,0-255,0-255
    tmp=cv2.cvtColor(HLS, cv2.COLOR_HLS2RGB)/255. #float32 in ,output float32 range[0,1] #int in, output 0-255
    # print('cv2RGB type',tmp.dtype,' max=',np.max(np.max(tmp,axis=0),axis=0),' min= ',np.min(np.min(tmp,axis=0),axis=0))
    # input()
    img=torch.tensor(tmp).to(torch.float).permute(2,0,1).unsqueeze(0)
    # hog_mask_rgb = cv2.cvtColor(hog_mask, cv2.COLOR_GRAY2RGB)
    return img
def rgba2rgb( rgba, background=(0,0,0) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background
    # rgb = (rgba[:,:,:3] * a + (1.0 - a) * rgba[:,:,:3]) / 255.0
    rgb[:,:,0] = (r * a + (1.0 - a) * R) / 255.0
    rgb[:,:,1] = (g * a + (1.0 - a) * G) / 255.0
    rgb[:,:,2] = (b * a + (1.0 - a) * B) / 255.0

    # return np.asarray( rgb, dtype='uint8' )
    return rgb   #Type is float32 [0,1]
def get_targetArea(img):
    if torch.cuda.is_available():
        img=img.permute(1,2,0).cpu().detach().numpy()
    else:
        img=img.permute(1,2,0).numpy()
    img = np.uint8(255*img)
    # img = np.array(remove(img),dtype=float).convert('RGB')
    img = rgba2rgb(remove(img)) #224,224,3   #Type is float32 [0,1]
    img = np.transpose(img,(2,0,1)) #3,224,224
    # print('img type:',type(img),' ',img)
    # print('img dimension by rembg:',img.shape)
   
    
    # input()
    return img
# def saveList(myList,filename):
#     # the filename should mention the extension 'npy'
#     np.save(filename,myList)
#     print("Saved successfully!")
# def loadList(filename):
#     # the filename should mention the extension 'npy'
#     tempNumpyArray=np.load(filename)
#     return tempNumpyArray.tolist()

if __name__ == "__main__":
    train_loader,_ = get_dataset(1,0)
    images, labels = next(iter(train_loader))
    image = images[0]
    print(time.ctime(time.time())) 
    image = get_targetArea(image)
    print(time.ctime(time.time())) 
    save_path = os.path.join(pathToFigure, 'rmbg.jpg')
    cv2.imwrite(save_path,image)