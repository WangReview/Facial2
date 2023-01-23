# -*- coding:utf-8 -*-
"""
作者：wwm
日期：2023年01月12日
"""

import os
import torch

def save(net, logger, hps, epoch):
    # create the path the checkpoint will be saved at using the epoch number
    path = os.path.join(hps['model_save_dir'], 'epoch_'+str(epoch))

    # create a dictionary containing the logger info and model info that will be saved
    checkpoint = {
        'logs': logger.get_logs(),
        'params': net.state_dict()
    }

    # save checkpoint
    torch.save(checkpoint, path)






