# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Options
import os, sys, glob,tqdm,json,shutil
import numpy as np
import random,time
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__) ))))
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import sparseconvnet as scn
import torch#, data_gen, np_ioueval
from SpSNet.core import suqeeze_model_v13_5 as suqeeze_model
from SpSNet.dataset import data_muti_frame_test as data
from SpSNet.utils import np_ioueval
from SpSNet.utils import utils

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = torch.cuda.is_available()
evealer = np_ioueval.evaler
device = torch.device('cuda' if use_cuda else 'cpu')
update = 1

unet=suqeeze_model.Model()
if use_cuda:
    unet=unet.cuda()

log_path = os.path.join(os.path.dirname(__file__) , 'snap')
log_dir = os.path.join(log_path,'unetv2_scale20_m16_rep1_v2')
snap = glob.glob(os.path.join(log_dir,'net*.pth'))
snap = sorted(snap,key= lambda x : int(x.split('-')[-1].split('.')[0]))
snap_num =32
if snap_num:
    snap = [ x for x in snap if str(snap_num) in x.split('/')[-1]]
# optimizer = optim.Adam(unet.parameters())
# print('#classifer parameters', sum([x.nelement() for x in unet.parameters()]))
base_dir = data.input_base
out_dit = os.path.join(data.data_base,'pre_muti_v13_5')
if not os.path.exists(out_dit):
    os.mkdir(out_dit)
out_dit_pre = os.path.join(out_dit,'sequences')
out_dit_prob = os.path.join(out_dit,'sequences_prob')
if not os.path.exists(out_dit_pre):
    os.mkdir(out_dit_pre)
if not os.path.exists(out_dit_prob):
    os.mkdir(out_dit_prob)
sq_pre_format = os.path.join(out_dit_pre,'{:02d}')
sq_prob_format = os.path.join(out_dit_prob,'{:02d}')
log_file_format  = 'predictions/{}.label'
log_prob_file_format  = 'predictions_prob/{}.npy'
sqs = []
files = []
for sq , x , y,_ in data.test:
    files.append((sq , x[-1].split('/')[-1].split('.')[0] ))
    sqs.append(sq)
sqs = set (sqs)
for x in sqs:
    p = sq_pre_format.format(x)
    pp = sq_prob_format.format(x)
    if not update:
        if os.path.exists(p):
            shutil.rmtree(p)
        if os.path.exists(pp):
            shutil.rmtree(pp)
        if os.path.exists(os.path.join(p,'predictions')):
            shutil.rmtree(os.path.join(p,'predictions'))
        if os.path.exists(os.path.join(pp,'predictions_prob')):
            shutil.rmtree(os.path.join(pp,'predictions_prob'))
    if not os.path.exists(p):
        os.mkdir(p)
    if not os.path.exists(pp):
        os.mkdir(pp)
    if not os.path.exists(os.path.join(p, 'predictions')):
        os.mkdir(os.path.join(p, 'predictions'))
    if not os.path.exists(os.path.join(pp, 'predictions_prob')):
        os.mkdir(os.path.join(pp, 'predictions_prob'))

for s in [snap[-1]]:

    start = time.time()
    unet.load_state_dict(torch.load(s))
    if True:
    # if scn.is_power2(epoch):
        with torch.no_grad():
            unet.eval()
            torch.cuda.empty_cache()
            scn.forward_pass_multiplyAdd_count=0
            scn.forward_pass_hidden_states=0
            for rep in range(1,1+data.val_reps):
                # times = time.time()
                for i,batch in tqdm.tqdm(enumerate(data.get_val_data_loader()),total= len(data.test)//(batch_size)):#
                    # print('one batch load: {}'.format(time.time() - times))
                    # times = time.time()
                    if use_cuda:
                        batch['x'][1] = [x.cuda() for x in batch['x'][1]]
                        idx = batch['id']
                        lp =  batch['lp']
                    predictions=unet(batch['x'])
                    start = 0
                    # print('one batch cal: {}'.format(time.time() - times))
                    # times = time.time()
                    for bid,id in enumerate(idx):
                        file = files[id]
                        prediction = predictions[start:start + batch['length'][bid]]
                        log_prob_file = os.path.join(sq_prob_format.format(file[0]),log_prob_file_format.format(file[1]))
                        log_file = os.path.join(sq_pre_format.format(file[0]),log_file_format.format(file[1]))
                        if os.path.exists(log_prob_file):
                            pre_prediction_prob = np.load(log_prob_file).astype(np.float64)
                        else:
                            pre_prediction_prob = np.zeros([lp[bid],np_ioueval.N_CLASSES],dtype=np.float64)
                        pre_prediction_prob[batch['point_ids'][bid]] += prediction.cpu().data.numpy()
                        pre_prediction = np.argmax(pre_prediction_prob,axis=-1)
                        np.save(log_prob_file,pre_prediction_prob )
                        # np.save(log_file,pre_prediction )
                        tmp = np.zeros_like(pre_prediction)
                        for k, v in data.config['learning_map_inv'].items():
                            tmp[pre_prediction == k] = v
                        pre_prediction = tmp
                        pre_prediction.astype(np.uint32).tofile(log_file)
                        start += batch['length'][bid]
                    # print('one batch save: {}'.format( time.time()- times ))
                    # times = time.time()
                # break
