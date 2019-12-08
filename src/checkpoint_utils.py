"""
Collection of utility functions relating to checkpointed models.
"""

import os
import torch


## NOTE see https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610
def checkpoint(model, optimizer=None, scheduler=None,
               epoch=None, checkpoint_dir=None, name=None):

    checkpoint_path = '{}checkpoint_{:03}.pth.tar'.format(
                        '' if name is None else (name + '.'), epoch)
    if checkpoint_dir is not None:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)

    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, checkpoint_path)
    print('\t=> Saved checkpoint file to "{}"'.format(checkpoint_path))
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    print('Loaded checkpoint from path "{}" (at epoch {})'
          .format(checkpoint_path, checkpoint['epoch']))
