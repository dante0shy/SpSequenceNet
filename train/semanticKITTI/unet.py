# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Options
import os, sys, glob,tqdm,json,yaml
import numpy as np
import random,time
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__) ))))
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import sparseconvnet as scn
import torch
from SpSNet.core import network as suqeeze_model
from SpSNet.dataset import data_muti_frame as data
from SpSNet.utils import np_ioueval
from SpSNet import config
import torch.optim as optim

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
use_cuda = torch.cuda.is_available()
evealer = np_ioueval.iouEval( config.N_CLASSES, ignore=config.UNKNOWN_ID)
device = torch.device('cuda' if use_cuda else 'cpu')

config_pos = os.path.dirname(__file__)
config_m = yaml.load(open(os.path.join(config_pos,'config.yaml')))


log_path = os.path.dirname(__file__) if 'log_pos' not in config_m['data'] else config
log_path = os.path.join(log_path, 'snap')
if not os.path.exists(log_path):
    os.mkdir(log_path)

exp_name='unetv2_scale{}_m{}_v2'.format( config_m['data']['scale'],config_m['model']['m'])
log_dir = os.path.join(log_path,exp_name)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
if os.path.isfile(os.path.join(log_dir,'log.josn')):
    log = json.load(open(os.path.join(log_dir,'log.josn'),'r'))
else:
    log = {}
snap = glob.glob(os.path.join(log_dir,'net*.pth'))
snap = list(sorted(snap,key=lambda x: int(x.split('-')[-1].split('.')[0])))[:34]

unet=suqeeze_model.Model(config.N_CLASSES,  config_m['data']['full_scale'],config_m['model']['m'])
if use_cuda:
    unet=unet.cuda()
# print(unet)

training_epochs=60
optimizer = optim.Adam(unet.parameters())

# '09d-'%epoch+'.pth'
epoch_s = 0
train_first = False
if snap:
    print('Restore from ' + snap[-1])
    unet.load_state_dict(torch.load(snap[-1]))
    epoch_s = int(snap[-1].split('-')[-1].split('.')[0])
    optimizer.load_state_dict(torch.load(snap[-1].replace('net-','optim-')))
    train_first = False #epoch_s == 0

training_epoch=60

print('#classifer parameters', sum([x.nelement() for x in unet.parameters()]))

dataloader = data.Dataloader(
    data_base = config_m['data']['data_base'],
    val_base= config_m['data']['val_base'],
    frames= data.frames,
    batch_size = config_m['data']['batch_size'],
    scale = config_m['data']['scale'],
    full_scale= config_m['data']['full_scale'], )


for epoch in range(epoch_s, training_epoch):
    unet.train()
    stats = {}
    start = time.time()
    train_loss=0
    mid = time.time()
    if epoch not in log.keys():
        log[epoch] = {
            'epoch' : epoch,
            'TrainLoss' : 0,
            'mIoU' : 0,
            'ValData' : ''
        }
    if train_first:#True:#
        print('train step : {}'.format(epoch))
        with tqdm.tqdm(total= len(dataloader.data_dict['train'])//dataloader.batch_size) as pbar:
            for i,batch in enumerate(dataloader.get_data_loader('train')):
                mid = time.time()
                optimizer.zero_grad()
                if use_cuda:
                    batch['x'][1] = [x.cuda() for x in batch['x'][1]]
                    batch['y'] = batch['y'].cuda()
                predictions = unet(batch['x'])
                loss = torch.nn.functional.cross_entropy(predictions, batch['y'])
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                pbar.set_postfix({"Train loss": "{0:1.5f}".format(train_loss/(i+1)),})
                pbar.update(1)
        print(epoch,'Train loss',train_loss/(i+1),'time=',time.time() - start,'s')
        torch.save(unet.state_dict(), os.path.join(log_dir,'net-%09d'%epoch+'.pth'))
        torch.save(optimizer.state_dict(), os.path.join(log_dir,'optim-%09d'%epoch+'.pth'))
        log[epoch]['TrainLoss'] = train_loss/(i+1)
        json.dump(log, open(os.path.join(log_dir, 'log.josn'), 'w'))
    else:
        train_first = True
    if True:
        # if not epoch % 2:
        # if scn.is_power2(epoch):
        print('val step : {}'.format(epoch))

        with torch.no_grad():
            unet.eval()
            scn.forward_pass_multiplyAdd_count = 0
            scn.forward_pass_hidden_states = 0
            start = time.time()
            evealer.reset()
            for i, batch in tqdm.tqdm(enumerate(dataloader.get_data_loader('valid'))):
                if use_cuda:
                    batch['x'][1] = [x.cuda() for x in batch['x'][1]]
                    batch['y'] = batch['y'].cuda()
                predictions = unet(batch['x'])
                evealer.addBatch(predictions.max(1)[1].cpu().numpy(), batch['y'].cpu().numpy())  # TLabel
                    # break
            print('epoch : ', epoch, 'Val MegaMulAdd=',
                      scn.forward_pass_multiplyAdd_count / len(data.val) / 1e6, 'MegaHidden',
                      scn.forward_pass_hidden_states / len(data.val) / 1e6, 'time=', time.time() - start, 's')
            m_iou, iou = evealer.getIoU()
            log[epoch]['mIoU'] = m_iou
            tp, fp, fn = evealer.getStats()
            total = tp + fp + fn
            print('classes          IoU')
            print('----------------------------')
            for i in range(config.N_CLASSES):
                    label_name = config.CLASS_LABELS[i]
                    log[epoch]['ValData'] += '{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})\n'.format(label_name, iou[i],tp[i],total[i])
                    print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(label_name, iou[i],
                                                                           tp[i],
                                                                           total[i]))
            print('mean IOU', m_iou,'\n')
    json.dump(log,open(os.path.join(log_dir,'log.josn'),'w'))