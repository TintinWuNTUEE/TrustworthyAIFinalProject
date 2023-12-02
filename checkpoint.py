import torch
import os 
from glob import glob

def _remove_recursively(folder_path):
  '''
  Remove directory recursively
  '''
  if os.path.isdir(folder_path):
    filelist = [f for f in os.listdir(folder_path)]
    for f in filelist:
      os.remove(os.path.join(folder_path, f))
  return
def _create_directory(directory):
  '''
  Create directory if doesn't exists
  '''
  if not os.path.exists(directory):
    os.makedirs(directory)
  return

def load_model(model,path,model_name):
    '''
    Load only model
    '''
    if os.listdir(path):
        file_path = sorted(glob(os.path.join(path, '*.pth')))[0]
        assert os.path.isfile(file_path), '=> No checkpoint found at {}'.format(path)
        checkpoint = torch.load(file_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    else:
        print('=> No checkpoint. Initializing model from scratch')
        os.makedirs(path)
        os.makedirs(os.path.join(path,model_name))
    return model

def load_checkpoint(model,model_name, optimizer, scheduler, path, logger,device):
    '''
    Load checkpoint file
    '''

    # if os.listdir(path) and os.listdir(os.path.join(path,model_name)):
    filepath=os.path.join(path,model_name)
    if os.listdir(filepath):
        file_path = sorted(glob(os.path.join(filepath, '*.pth')))[0]
        print("model file path:",file_path)
        # checkpoint = torch.load(file_path, map_location='cpu') 
        checkpoint = torch.load(file_path, map_location=device) 
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint.pop('startEpoch')
        optimizer.load_state_dict(checkpoint.pop('optimizer'))
        scheduler.load_state_dict(checkpoint.pop('scheduler'))
        logger.info('Checkpoint loaded at {}'.format(file_path))
        return model, optimizer, scheduler, epoch
    else:
        print('=> No checkpoint. Initializing model from scratch')
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(os.path.join(path,model_name)):
          os.makedirs(os.path.join(path,model_name))
        epoch = 1
    return model, optimizer, scheduler, epoch

def save_checkpoint(path,model_name,model,optimizer,epoch,scheduler,scaler):
    '''
    Save checkpoint file
    '''
    path = os.path.join(path, model_name)

  # Remove recursively if epoch_last folder exists and create new one
    _remove_recursively(path)
    _create_directory(path)

    weights_fpath = os.path.join(path ,'epoch_{}.pth'.format(str(epoch).zfill(3)))

    torch.save({
    'startEpoch': epoch+1,  # To start on next epoch when loading the dict...
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    "scaler": scaler.state_dict()
  }, weights_fpath)

    return weights_fpath
    