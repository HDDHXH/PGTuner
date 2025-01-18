# -*- coding: utf-8 -*-
"""

Prioritized Replay Memory
"""
import random
import pickle
import numpy as np


class SumTree(object):
    '''
    这段代码定义了一个名为 SumTree 的类，这是一种特殊的树结构，常用于强化学习中的优先级回放队列（Priority Replay Buffer）
    '''

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.num_entries = 0
        self.write = 0

    def _propagate(self, idx, change):  #递归方法，用于在树中更新路径上的节点值。当一个叶节点的值被更新时，这个变化（change）需要沿着树向上传播至根
        parent = (idx - 1) // 2

        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s): #递归方法，用于根据一个随机值 s 找到对应的叶节点
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self): #返回树的根节点值，即所有叶节点值的总和
        return self.tree[0]

    def add(self, p, data): #添加数据到树中
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.num_entries < self.capacity:
            self.num_entries += 1

    def update(self, idx, p):  #更新叶节点的值并传播这一变化
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s): #根据一个随机值 s 获取相应的数据
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return [idx, self.tree[idx], self.data[data_idx]]


class PrioritizedReplayMemory(object):
    '''
        这段代码定义了一个名为 PrioritizedReplayMemory 的类，该类实现了优先级回放缓冲区，主要用于强化学习。
        这个缓冲区使用之前定义的 SumTree 类来存储经验样本，这些样本根据它们的优先级（错误大小）进行存储和抽样
        '''

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.e = 0.01 #一个小的正数，避免优先级为零
        self.a = 0.6 #用于计算优先级的指数，影响优先级分布的形状
        self.beta = 0.4 #用于计算重要性采样权重（Importance Sampling Weights）的指数
        self.beta_increment_per_sampling = 0.001 #每次采样后 beta 的增量，直到最大为1

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        # (s, a, r, s, t)
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def __len__(self):
        return self.tree.num_entries

    def sample(self, n):
        batch = []
        idxs = []

        segment = self.tree.total() / n
        # priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            # priorities.append(p)
            
            batch.append(data)
            idxs.append(idx)

        return batch, idxs  #是一个列表

        # sampling_probabilities = priorities / self.tree.total()
        # is_weight = np.power(self.tree.num_entries * sampling_probabilities, -self.beta)
        # is_weight /= is_weight.max()

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def save(self, path):
        f = open(path, 'wb')
        pickle.dump({"tree": self.tree}, f)
        f.close()

    def load_memory(self, path):
        with open(path, 'rb') as f:
            _memory = pickle.load(f)
        self.tree = _memory['tree']

