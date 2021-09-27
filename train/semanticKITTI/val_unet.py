# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Options
import os, sys, glob, tqdm, json, shutil, yaml
import numpy as np
import random, time

print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import sparseconvnet as scn
import torch  # , data_gen, np_ioueval
from SpSNet.core import network as suqeeze_model
from SpSNet.dataset import data_muti_frame as data
from SpSNet.utils import np_ioueval
from SpSNet import config

config_pos = os.path.dirname(__file__)
config_m = yaml.load(open(os.path.join(config_pos,'config.yaml')))

os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(config_m['CUDA_VISIBLE_DEVICES'])

use_cuda = torch.cuda.is_available()

evealer = np_ioueval.iouEval( config.N_CLASSES, ignore=config.UNKNOWN_ID)
device = torch.device("cuda" if use_cuda else "cpu")
update = 1

unet=suqeeze_model.Model(config.N_CLASSES,  config_m['data']['full_scale'],config_m['model']['m'])
if use_cuda:
    unet = unet.cuda()

log_path = (
    os.path.dirname(__file__)
    if "log_pos" not in config_m["data"]
    else config_m["data"]["log_pos"]
)

if 'val_model_dir' not in  config_m['model']:
    log_path = os.path.join(log_path, "snap")
    log_dir = os.path.join(
        log_path,
        "unetv2_scale{}_m{}_v2".format(config_m["data"]["scale"], config_m["model"]["m"]),
    )
else:
    log_path = config_m['model']['val_model_dir']
    log_dir= log_path

snap = glob.glob(os.path.join(log_dir, "net*.pth"))
snap = sorted(snap, key=lambda x: int(x.split("-")[-1].split(".")[0]))

dataloader = data.Dataloader(
    data_base=config_m["data"]["data_base"],
    val_base=config_m["data"]["val_base"],
    frames=data.frames,
    batch_size=config_m["data"]["batch_size"],
    scale=config_m["data"]["scale"],
    full_scale=config_m["data"]["full_scale"],
)

base_dir = config_m["data"]["val_base"]
out_dit = os.path.join(base_dir, "val_muti_v1")
if not os.path.exists(out_dit):
    os.mkdir(out_dit)
out_dit_pre = os.path.join(out_dit, "sequences")
out_dit_prob = os.path.join(out_dit, "sequences_prob")
if not os.path.exists(out_dit_pre):
    os.mkdir(out_dit_pre)
if not os.path.exists(out_dit_prob):
    os.mkdir(out_dit_prob)
sq_pre_format = os.path.join(out_dit_pre, "{:02d}")
sq_prob_format = os.path.join(out_dit_prob, "{:02d}")
log_file_format = "predictions/{}.label"
log_prob_file_format = "predictions_prob/{}.npy"
sqs = []
files = []
for sq, x, y, _ in dataloader.data_dict["valid"]:
    files.append((sq, x[-1].split("/")[-1].split(".")[0]))
    sqs.append(sq)
sqs = set(sqs)
for x in sqs:
    p = sq_pre_format.format(x)
    pp = sq_prob_format.format(x)
    if not update:
        if os.path.exists(p):
            shutil.rmtree(p)
        if os.path.exists(pp):
            shutil.rmtree(pp)
        if os.path.exists(os.path.join(p, "predictions")):
            shutil.rmtree(os.path.join(p, "predictions"))
        if os.path.exists(os.path.join(pp, "predictions_prob")):
            shutil.rmtree(os.path.join(pp, "predictions_prob"))
    if not os.path.exists(p):
        os.mkdir(p)
    if not os.path.exists(pp):
        os.mkdir(pp)
    if not os.path.exists(os.path.join(p, "predictions")):
        os.mkdir(os.path.join(p, "predictions"))
    if not os.path.exists(os.path.join(pp, "predictions_prob")):
        os.mkdir(os.path.join(pp, "predictions_prob"))



# for s in snap:
for _ in range(3):
    start = time.time()
    # print('loading: {}'.format(s))
    # unet.load_state_dict(torch.load(s))
    if True:
        # if scn.is_power2(epoch):
        with torch.no_grad():
            unet.eval()
            torch.cuda.empty_cache()
            scn.forward_pass_multiplyAdd_count = 0
            scn.forward_pass_hidden_states = 0
            evealer.reset()
            for i, batch in tqdm.tqdm(
                enumerate(dataloader.get_data_loader("valid")),
                total=len(dataloader.data_dict["valid"]) // dataloader.batch_size,
            ):  #

                if use_cuda:
                    batch["x"][1] = [x.cuda() for x in batch["x"][1]]
                    idx = batch["id"]
                    lp = batch["lp"]
                predictions = unet(batch["x"])
                start = 0
                start_y = 0
                for bid, id in enumerate(idx):
                    file = files[id]
                    prediction = predictions[start : start + batch["length"][bid]]
                    log_prob_file = os.path.join(
                        sq_prob_format.format(file[0]),
                        log_prob_file_format.format(file[1]),
                    )
                    log_file = os.path.join(
                        sq_pre_format.format(file[0]), log_file_format.format(file[1])
                    )
                    if os.path.exists(log_prob_file):
                        pre_prediction_prob = np.load(log_prob_file).astype(np.float64)
                    else:
                        pre_prediction_prob = np.zeros(
                            [lp[bid], config.N_CLASSES], dtype=np.float64
                        )
                    pre_prediction_prob[
                        batch["point_ids"][bid]
                    ] += prediction.cpu().data.numpy()
                    pre_prediction = np.argmax(pre_prediction_prob, axis=-1)
                    evealer.addBatch(
                        pre_prediction, batch["y"][start_y : start_y + lp[bid]].numpy()
                    )  # TLabel
                    tmp = np.zeros_like(pre_prediction)
                    for k, v in dataloader.config["learning_map_inv"].items():
                        tmp[pre_prediction == k] = v
                    pre_prediction = tmp
                    pre_prediction.astype(np.uint32).tofile(log_file)
                    start += batch["length"][bid]
                    start_y += lp[bid]
            print(
                "Val MegaMulAdd=",
                scn.forward_pass_multiplyAdd_count / len(dataloader.data_dict["valid"]) / 1e6,
                "MegaHidden",
                scn.forward_pass_hidden_states / len(dataloader.data_dict["valid"]) / 1e6,
                "time=",
                time.time() - start,
                "s",
            )
            m_iou, iou = evealer.getIoU()

            tp, fp, fn = evealer.getStats()
            total = tp + fp + fn
            print("classes          IoU")
            print("----------------------------")
            for i in range(config.N_CLASSES):
                label_name = config.CLASS_LABELS[i]
                print(
                    "{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})".format(
                        label_name, iou[i], tp[i], total[i]
                    )
                )
            print("mean IOU", m_iou, "\n")
