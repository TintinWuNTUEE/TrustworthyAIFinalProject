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

def load_model(model,path):
    '''
    Load only model
    '''
    if os.listdir(path):
        file_path = sorted(glob(os.path.join(path, '*.pth')))[0]
        assert os.path.isfile(file_path), '=> No checkpoint found at {}'.format(path)
        checkpoint = torch.load(file_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    return model

def load_checkpoint(model, optimizer, scheduler,scaler, path, logger):
    '''
    Load checkpoint file
    '''

    if os.listdir(path):
        file_path = sorted(glob(os.path.join(path, '*.pth')))[0]
        checkpoint = torch.load(file_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint.pop('startEpoch')
        optimizer.load_state_dict(checkpoint.pop('optimizer'))
        scheduler.load_state_dict(checkpoint.pop('scheduler'))
        scaler.load_state_dict(checkpoint["scaler"])
        logger.info('=> Continuing training routine. Checkpoint loaded at {}'.format(file_path))
        return model, optimizer, scheduler,scaler, epoch
    else:
        logger.info('=> No checkpoint. Initializing model from scratch')
        epoch = 1
    return model, optimizer, scheduler,scaler, epoch

def save_checkpoint(path,model,optimizer,epoch,scheduler,scaler):
    '''
    Save checkpoint file
    '''

  # Remove recursively if epoch_last folder exists and create new one
    _remove_recursively(path)
    _create_directory(path)

    weights_fpath = os.path.join(path, 'epoch_{}.pth'.format(str(epoch).zfill(3)))

    torch.save({
    'startEpoch': epoch+1,  # To start on next epoch when loading the dict...
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    "scaler": scaler.state_dict()
  }, weights_fpath)

    return weights_fpath
    