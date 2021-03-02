# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Options
import torch, numpy as np, glob, math, torch.utils.data
import os,time,yaml,tqdm,random
from SpSNet.utils.generate_sequential import *

frames = 2
class Dataloader():

    def __init__(self,data_base, val_base, frames, batch_size, scale, full_scale):
        self.frames = frames
        self.scale = scale
        self.full_scale = full_scale
        self.batch_size = batch_size
        self.data_base =data_base
        self.val_base =val_base

        self.input_base = os.path.join(data_base,'sequences')
        self.config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__),'config/semantic-kitti-all.yaml')))
        dirs = glob.glob(os.path.join(self.input_base,'*'))
        self.data_dict = self.get_dataset(dirs)

    def get_dataset(self,datadirs):
        files = {
            'train':[],
            'valid':[],
            'test':[]
        }
        for dir_ in datadirs:
            datas = glob.glob(os.path.join(dir_, 'velodyne', '*'))
            labels = glob.glob(os.path.join(dir_, 'labels', '*'))
            times = os.path.join(dir_, 'times.txt')
            clib = parse_calibration(os.path.join(dir_, 'calib.txt'))
            poses = parse_poses(os.path.join(dir_, 'poses.txt'),clib)

            # labels_dict = {x[-12:-6]: x for x in labels}
            sq = int(dir_[-2:])
            split = [k for k, v in self.config['split'].items() if sq in v][0]
            if split not in files.keys():
                continue
            # f = [(sq, data, labels_dict[data[-10:-4]] if len(labels_dict) else '') for data in datas]
            datas = list(sorted(datas,key= lambda x : int(x[-10:-4])))
            labels = list(sorted(labels,key= lambda x : int(x[-12:-6])))
            f = []

            for i , data in enumerate(datas):
                pre = i-frames+1 if i-frames+1>=0 else 0
                data_frames = datas[ pre:i+1]
                pose_frames = poses[pre:i+1]
                if split == 'test':
                    label_frames =['']*min(i+1,frames)
                else:
                    label_frames = labels[pre:i+1]

                f.append((sq,data_frames,label_frames,pose_frames))
            files[split].extend(f)
        return  files


    def _get(self,tbl,mode = 'train'):
        locs=[]
        feats=[]
        labels=[]
        for idx,i in enumerate(tbl):
            locs_pre = []
            feats_pre = []
            labels_pre = []

            scans_b = self.data_dict[mode][i][1]
            labels_b = self.data_dict[mode][i][2]
            poss_b = self.data_dict[mode][i][3]
            pre_frame = len(scans_b)-1

            if mode == 'train':
                seed = np.eye(3) + np.random.randn(3, 3) * 0.01
                seed[0][0] *= np.random.randint(0, 2) * 2 - 1
                theta = np.random.rand() * 2 * math.pi
                seed = np.matmul(seed, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
            else:
                seed = np.eye(3)

            def get_data(frame_idx,mean = None):
                scan = np.fromfile(scans_b[frame_idx], dtype=np.float32)
                scan = scan.reshape((-1, 4))
                r = scan[:, 3]

                if frame_idx != pre_frame:
                    scan[:,3] = 1
                    diff = np.matmul(inv(poss_b[-1]), poss_b[frame_idx])
                    scan = np.matmul(diff, scan.T).T
                coords = scan[:, :3]
                try:
                    mean[0]
                except:
                    mean = coords.mean(0)
                coords = coords - mean
                coords = self.aug_rotate(coords*self.scale,seed)
                # coords = self.aug_rotate(coords*self.scale,seed)
                idxs = (coords.min(1) >= 0) * (coords.max(1) < self.full_scale)
                coords = coords[idxs]
                r = r[idxs]
                coords = torch.from_numpy(coords).long()
                if frame_idx== pre_frame:
                    label = np.fromfile(labels_b[frame_idx], dtype=np.uint32)
                    sem_label = label & 0xFFFF
                    for k, v in self.config['learning_map'].items():
                        sem_label[sem_label == k] = v
                    sem_label = sem_label[idxs]
                    return coords,r,sem_label,mean
                else:
                    if coords.shape[0]:
                        return coords, r,[0],mean
                    else:
                        return torch.from_numpy(np.array([[0,0,0]])).long(),np.array([[0,]]),[0],mean
            tmp_pre =get_data(pre_frame)
            locs_pre.append(torch.cat([tmp_pre[0], torch.LongTensor(tmp_pre[0].shape[0], 1).fill_(idx)], 1))
            feats_pre.append(torch.from_numpy(tmp_pre[1].reshape(tmp_pre[0].shape[0], 1)) .float()+ torch.randn(1) * 0.1)
            labels_pre.append(torch.from_numpy(tmp_pre[2].astype(np.int32)))

            if pre_frame:
                tmp = list(map(lambda x : get_data(x,tmp_pre[-1]),range(pre_frame)))
            else:
                tmp = []

            if tmp:
                a,b,c = [],[],[]
                for d in tmp:
                    a.append(d[0])
                    b.append(d[1])
            for i in range(self.frames-1):
                pos =  pre_frame-i-1
                if pos >=0:
                    locs_pre.append(torch.cat([a[pos],torch.LongTensor(a[pos].shape[0],1).fill_(idx)],1))
                    feats_pre.append(torch.from_numpy(b[pos].reshape(a[pos].shape[0],1)).float()+torch.randn(1)*0.1)
                    # labels.append(torch.from_numpy(c[pos].astype(np.int32)))
                else:
                    locs_pre.append(torch.LongTensor([[0,0,0,idx]]))
                    feats_pre.append(torch.from_numpy(np.array([[0.]])).float())
            locs.append(locs_pre)
            feats.append(feats_pre)
            labels.extend(labels_pre)
        locs = list(zip(*locs))
        locs=[torch.cat(loc,0) for loc in locs]
        feats=list(zip(*feats))
        feats=[torch.cat(feat,0) for feat in feats]
        labels=torch.cat(labels,0)
        return {'x': [locs,feats], 'y': labels.long(), 'id': tbl}

    def get_data_loader(self,mode):
        assert  mode in ['train','test','valid']
        data = self.data_dict['train']
        return torch.utils.data.DataLoader(
            list(range(len(data))),batch_size=self.batch_size , collate_fn=lambda x: self._get(x,mode), num_workers=20, shuffle=mode in ['train'])

    def aug_rotate(self,a,seed):
        a = np.matmul(a, seed)+self.full_scale/2
        return a[:, :3]

if __name__=='__main__':
    pass