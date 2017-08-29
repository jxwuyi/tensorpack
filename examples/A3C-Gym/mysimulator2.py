#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: simulator.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import multiprocessing as mp
import time
import os
import threading
from abc import abstractmethod, ABCMeta
from collections import defaultdict

import six
from six.moves import queue
import zmq

from tensorpack.callbacks import Callback
from tensorpack.tfutils.varmanip import SessionUpdate
from tensorpack.predict import OfflinePredictor
from tensorpack.utils import logger
from tensorpack.utils.serialize import loads, dumps
from tensorpack.utils.concurrency import LoopThread, ensure_proc_terminate

import sys
import torch
import torch.nn.functional as F
from torch.autograd import Variable
path_to_python_repo = '/home/jxwuyi/workspace/objrender/python'
sys.path.insert(0, path_to_python_repo)
colorFile = '/home/jxwuyi/workspace/objrender/metadata/colormap_coarse.csv'
csvFile = '/home/jxwuyi/data/fb/data/metadata/ModelCategoryMapping.csv'
prefix = '/home/jxwuyi/data/fb/data/house/'
all_houseIDs = ['00065ecbdd7300d35ef4328ffe871505',
'cf57359cd8603c3d9149445fb4040d90', 'ff32675f2527275171555259b4a1b3c3', '775941abe94306edc1b5820e3a992d75',
'7995c2a93311717a3a9c48d789563590', '8b8c1994f3286bfc444a7527ffacde86', '31966fdc9f9c87862989fae8ae906295',
'32e53679b33adfcc5a5660b8c758cc96', '4383029c98c14177640267bd34ad2f3c', '0884337c703e7c25949d3a237101f060',
'492c5839f8a534a673c92912aedc7b63', 'a7e248efcdb6040c92ac0cdc3b2351a6', '2364b7dcc432c6d6dcc59dba617b5f4b',
'e3ae3f7b32cf99b29d3c8681ec3be321', 'f10ce4008da194626f38f937fb9c1a03', 'e6f24af5f87558d31db17b86fe269cf2',
'1dba3a1039c6ec1a3c141a1cb0ad0757', 'b814705bc93d428507a516b866efda28', '26e33980e4b4345587d6278460746ec4',
'5f3f959c7b3e6f091898caa8e828f110', 'b5bd72478fce2a2dbd1beb1baca48abd', '9be4c7bee6c0ba81936ab0e757ab3d61']
import environment
from environment import SimpleHouseEnv as HouseEnv
from world import World

def genCacheFile(houseID):
    return prefix + houseID + '/cachedmap1k.pkl'

def create_world(houseID):
    objFile = prefix + houseID + '/house.obj'
    jsonFile = prefix + houseID + '/house.json'
    cachedFile = genCacheFile(houseID)
    assert os.path.isfile(cachedFile), '[Warning] No Cached Map File Found for House <{}>!'.format(houseID)
    world = World(jsonFile, objFile, csvFile, 1000,
                  CachedFile=cachedFile, EagleViewRes=100)
    return world

class NaiveHouseEnvironment:
    def __init__(self, id = 0):
        self.world = create_world(all_houseIDs[id % 20])
        self.env = HouseEnv(self.world, colorFile, resolution=(120,90), linearReward=True,
                            hardness=0.6, action_degree=4,
                            segment_input=True,
                            use_segment_id=False,
                            joint_visual_signal=True)
        self.obs = self.env.reset()

    def current_state(self):
        return self.obs

    def action(self, act):
        obs, rew, done, info = self.env(act)
        if done:
            obs = self.env.reset()
        self.obs = obs
        return rew, done

import multiprocessing

n_proc = 5
device = 2

def worker(num):
    """thread worker function"""
    print('Start Proc%d ...' % num)
    per_proc_sample = 200
    world = create_world(all_houseIDs[num % 20])
    env = HouseEnv(world, colorFile, resolution=(120, 90), linearReward=True,
                   hardness=0.6, action_degree=4,
                   segment_input=True,
                   use_segment_id=False,
                   joint_visual_signal=True,
                   render_device=device)
    obs = env.reset()
    tstart = time.time()
    print('Proc<%d> Start Time = %.8f' % (num, tstart))
    import random
    for _ in range(per_proc_sample):
        act = random.randint(0, 12)
        _obs, rew, done, info = env.step(act)
        if done:
            obs = env.reset()
    det = time.time() - tstart
    print('>>> Proc<%d> Done! Total Sample = %d, Time Elapsed = %.5f, Sample per Sec = %.5f' % (num, per_proc_sample, det, per_proc_sample/det ))
    return

if __name__ == '__main__':
    jobs = []
    for i in range(n_proc):
        p = multiprocessing.Process(target=worker, args=(i,))
        jobs.append(p)
        p.start()

    time.sleep(10000)