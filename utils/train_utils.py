import copy
import os
import math

import torch
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
import matplotlib.pyplot as plt

def get_saved_state(model, optimizer, lr_scheduler, epoch, configs):
    """Get the information to save with checkpoints"""
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    utils_state_dict = {
        'epoch': epoch,
        'configs': configs,
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'lr_scheduler': copy.deepcopy(lr_scheduler.state_dict())
    }

    return model_state_dict, utils_state_dict


def save_checkpoint(checkpoints_dir, saved_fn, model_state_dict, utils_state_dict, epoch):
    """Save checkpoint every epoch only is best model or after every checkpoint_freq epoch"""
    model_save_path = 'checkpoints/Model_complexer_yolo_V4.pth'
    # model_save_path = 'checkpoints/Model_complexer_yolo_epoch_{}.pth'.format(str(epoch).zfill(3))
    # model_save_path = os.path.join(checkpoints_dir, 'Model_yolo3d_yolov4.pth')
    # model_save_path = os.path.join(checkpoints_dir, 'Model_{}_epoch_{}.pth'.format(saved_fn, epoch))
    # utils_save_path = os.path.join(checkpoints_dir, 'Utils_{}_epoch_{}.pth'.format(saved_fn, epoch))

    torch.save(model_state_dict, model_save_path)
    # torch.save(utils_state_dict, utils_save_path)

    print('save a checkpoint at {}'.format(model_save_path))


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def get_tensorboard_log(model):
    if hasattr(model, 'module'):
        yolo_layers = model.module.yolo_layers
    else:
        yolo_layers = model.yolo_layers

    tensorboard_log = {}
    tensorboard_log['Average_All_Layers'] = {}
    for idx, yolo_layer in enumerate(yolo_layers, start=1):
        layer_name = 'YOLO_Layer{}'.format(idx)
        tensorboard_log[layer_name] = {}
        for name, metric in yolo_layer.metrics.items():
            tensorboard_log[layer_name]['{}'.format(name)] = metric
            if idx == 1:
                tensorboard_log['Average_All_Layers']['{}'.format(name)] = metric / len(yolo_layers)
            else:
                tensorboard_log['Average_All_Layers']['{}'.format(name)] += metric / len(yolo_layers)

    return tensorboard_log


def plot_lr_scheduler(optimizer, scheduler, num_epochs=300, save_dir=''):
    # Plot LR simulating training for full num_epochs
    optimizer, scheduler = copy.copy(optimizer), copy.copy(scheduler)  # do not modify originals
    y = []
    for _ in range(num_epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, num_epochs)
    plt.ylim(0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'LR.png'), dpi=200)

