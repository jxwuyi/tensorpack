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

__all__ = ['SimulatorProcess', 'SimulatorMaster',
           'SimulatorProcessStateExchange',
           'TransitionExperience']


class TransitionExperience(object):
    """ A transition of state, or experience"""

    def __init__(self, state, action, reward, **kwargs):
        """ kwargs: whatever other attribute you want to save"""
        self.state = state
        self.action = action
        self.reward = reward
        for k, v in six.iteritems(kwargs):
            setattr(self, k, v)


@six.add_metaclass(ABCMeta)
class SimulatorProcessBase(mp.Process):
    def __init__(self, idx):
        super(SimulatorProcessBase, self).__init__()
        self.idx = int(idx)
        self.name = u'simulator-{}'.format(self.idx)
        self.identity = self.name.encode('utf-8')

    @abstractmethod
    def _build_player(self):
        pass


class SimulatorProcessStateExchange(SimulatorProcessBase):
    """
    A process that simulates a player and communicates to master to
    send states and receive the next action
    """

    def __init__(self, idx, pipe_c2s, pipe_s2c):
        """
        Args:
            idx: idx of this process
            pipe_c2s, pipe_s2c (str): name of the pipe
        """
        super(SimulatorProcessStateExchange, self).__init__(idx)
        self.c2s = pipe_c2s
        self.s2c = pipe_s2c

    def run(self):
        player = self._build_player()
        context = zmq.Context()
        c2s_socket = context.socket(zmq.PUSH)
        c2s_socket.setsockopt(zmq.IDENTITY, self.identity)
        c2s_socket.set_hwm(2)
        c2s_socket.connect(self.c2s)

        s2c_socket = context.socket(zmq.DEALER)
        s2c_socket.setsockopt(zmq.IDENTITY, self.identity)
        # s2c_socket.set_hwm(5)
        s2c_socket.connect(self.s2c)

        state = player.current_state()
        reward, isOver = 0, False
        while True:
            c2s_socket.send(dumps(
                (self.identity, state, reward, isOver)),
                copy=False)
            action = loads(s2c_socket.recv(copy=False).bytes)
            reward, isOver = player.action(action)
            state = player.current_state()


# compatibility
SimulatorProcess = SimulatorProcessStateExchange


class SimulatorMaster(threading.Thread):
    """ A base thread to communicate with all StateExchangeSimulatorProcess.
        It should produce action for each simulator, as well as
        defining callbacks when a transition or an episode is finished.
    """
    class ClientState(object):
        def __init__(self):
            self.memory = []    # list of Experience

    def __init__(self, pipe_c2s, pipe_s2c):
        super(SimulatorMaster, self).__init__()
        assert os.name != 'nt', "Doesn't support windows!"
        self.daemon = True
        self.name = 'SimulatorMaster'

        self.context = zmq.Context()

        self.c2s_socket = self.context.socket(zmq.PULL)
        self.c2s_socket.bind(pipe_c2s)
        self.c2s_socket.set_hwm(10)
        self.s2c_socket = self.context.socket(zmq.ROUTER)
        self.s2c_socket.bind(pipe_s2c)
        self.s2c_socket.set_hwm(10)

        # queueing messages to client
        self.send_queue = queue.Queue(maxsize=100)

        def f():
            msg = self.send_queue.get()
            self.s2c_socket.send_multipart(msg, copy=False)
        self.send_thread = LoopThread(f)
        self.send_thread.daemon = True
        self.send_thread.start()

        # make sure socket get closed at the end
        def clean_context(soks, context):
            for s in soks:
                s.close()
            context.term()
        import atexit
        atexit.register(clean_context, [self.c2s_socket, self.s2c_socket], self.context)

        self.cnt = 0
        self.start_time = time.time()

    def run(self):
        self.clients = defaultdict(self.ClientState)
        try:
            while True:
                msg = loads(self.c2s_socket.recv(copy=False).bytes)
                ident, state, reward, isOver = msg
                # TODO check history and warn about dead client
                client = self.clients[ident]

                # check if reward&isOver is valid
                # in the first message, only state is valid
                if len(client.memory) > 0:
                    client.memory[-1].reward = reward
                    if isOver:
                        self._on_episode_over(ident)
                    else:
                        self._on_datapoint(ident)
                # feed state and return action
                self._on_state(state, ident)
        except zmq.ContextTerminated:
            logger.info("[Simulator] Context was terminated.")

    @abstractmethod
    def _on_state(self, state, ident):
        """response to state sent by ident. Preferrably an async call"""

    @abstractmethod
    def _on_episode_over(self, client):
        """ callback when the client just finished an episode.
            You may want to clear the client's memory in this callback.
        """

    def _on_datapoint(self, client):
        """ callback when the client just finished a transition
        """

    def __del__(self):
        self.context.destroy(linger=0)


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

if __name__ == '__main__':
    class NaiveSimulator(SimulatorProcess):

        def _build_player(self):
            return NaiveHouseEnvironment()

    class NaiveActioner(SimulatorMaster):
        def _get_action(self, state):
            def gen_act(d):
                x = F.softmax(Variable(torch.randn(d)))
                return x.data.numpy()
            return [gen_act(4), gen_act(2)]

        def _on_episode_over(self, client):
            # print("Over: ", client.memory)
            client.memory = []
            client.state = 0

        def _on_state(self, state, ident):
            self.cnt += 1
            if self.cnt % 500 == 0:
                det = time.time() - self.start_time
                print('>> Samples = %d, Total Time Elapsed = %.4fs, Samples per Sec = %.5fs' % (self.cnt, det, self.cnt/det))

    name = 'ipc://whatever'
    name2 = 'ipc://whatever2'
    procs = [NaiveSimulator(k, name, name2) for k in range(20)]
    [k.start() for k in procs]

    th = NaiveActioner(name, name2)
    ensure_proc_terminate(procs)
    th.start()

    time.sleep(100)
