# -*- coding: utf-8 -*-

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.tensor as T
from random import randint
from param2vector import param2vector

'''
param_mode = 'export' : param_file write mode -> 'w'
param_mode = 'import' : param_file read mode -> 'r'
param_mode = 'IEport' : param_file upgrade mode -> '+'
'''
class MultipleRegression(object) :
    def __init__(self, batch, dataset, param, mode) :
        self.__mode = {'export': 'w', 'import': '+'}
        self.batch = batch
        self.param = param
        self.mode = self.__mode[mode]
        self.label = dataset[0]
        self.case = dataset[1]
        self.case_len = len(self.case)
        self.label_len = len(self.label)
        self.param_len = len(self.case[0])
        self.accuracy = 0
        self.learn_rate = 0.001

        if (self.case_len != self.label_len) :
            print('The length of case and label is not match!')
            return -1
        elif (self.batch > self.case_len) :
            print('Batch is longer than the length of case and label!')
            return -1
        else :
            self.input = t.ones(self.batch, self.param_len, requires_grad = False)
            self.right = t.ones(self.batch, 1, requires_grad = False)
            self.output = t.ones(self.batch, 1, requires_grad = True)

        if (self.mode == 'w') :
            self.weight = t.ones(self.param_len, 1, requires_grad = True)
        elif (self.mode == '+') :
            self.weight = t.load(self.param)

    def __get_batch(self) :
        for i in range(0, self.batch):
            random_id = randint(0, self.case_len - 1)
            self.input[i] = T(self.case[random_id])
            self.right[i] = T(self.label[random_id])

    def __loss(self) :
        self.loss = F.pairwise_distance(self.output, self.right, 2, eps = 1e-10, keepdim = True)

    def __loss_mean(self) :
        self.loss_mean = self.loss.mean()
        return self.loss_mean

    def __forward(self) :
        self.__get_batch()
        for j in range(0, self.batch):
            self.output = F.relu(self.input.mm(self.weight))
            # self.output = self.input.mm(self.weight)
            self.__loss()

    def __upgrade_grad(self) :
        with t.no_grad() :
            self.weight -= self.weight.grad * self.learn_rate

    def __zero_grad(self) :
        with t.no_grad() :
            self.weight.grad.zero_()

    def __upgrade_weight(self) :
        self.__upgrade_grad()
        self.__zero_grad()

    def __accuracy(self) :
        rate = t.div(self.loss, self.right)
        for i in range(0, self.batch) :
            self.accuracy += (rate[i][0] <= 0.2) and 1 or 0
        self.accuracy = self.accuracy / self.batch

    def __weight2file(self) :
        t.save(self.weight, self.param)

    def BackWard(self, iteration, learn_rate) :
        self.iteration = iteration
        self.learn_rate = learn_rate
        for i in range(0, self.iteration) :
            self.__get_batch()
            self.__forward()
            self.__loss_mean().backward()
            self.__upgrade_weight()
            self.__accuracy()
            self.__weight2file()
            print('iteration:', i, 'loss_mean:', self.loss_mean.data, 'accuracy:%.4f' % self.accuracy)

    def WhileBackWard(self, accuracy, learn_rate) :
        key = 0
        while (key < 1000) :
            self.BackWard(1, learn_rate)
            if (self.accuracy >= accuracy):
                key += 1
            else:
                key = 1

    def BestWhileBackWard(self, accuracy) :
        key = 0
        learn_rate = 0
        while (key < 1000) :
            if (self.accuracy > 0.1) :
                learn_rate = 0.0005
            elif (self.accuracy > 0.2) :
                learn_rate = 0.0001
            elif (self.accuracy > 0.3) :
                learn_rate = 0.00005
            elif (self.accuracy > 0.4) :
                learn_rate = 0.00001
            elif (self.accuracy > 0.5) :
                learn_rate = 0.000005
            elif (self.accuracy > 0.6) :
                learn_rate = 0.000001
            elif (self.accuracy > 0.7) :
                learn_rate = 0.0000005
            elif (self.accuracy > 0.8) :
                learn_rate = 0.0000001
            elif (self.accuracy > 0.9) :
                learn_rate = 0.00000001
            else :
                learn_rate = 0.001

            self.BackWard(1, learn_rate)

            if (self.accuracy >= accuracy):
                key += 1
            else:
                key = 0

    def printdata(self):
        self.BackWard(1, self.learn_rate)
        self.__upgrade_grad()
        print('input:', self.input.data)
        print('weight:', self.weight.data)
        print('weight.grad:', self.weight.grad)
        print('output:', self.output.data)
        print('right:', self.right.data)
        self.__zero_grad()
# example
dataset = param2vector('./testcase.param')
a = MultipleRegression(20, dataset, './param.bin', 'import')
# a.BackWard(10000, 0.001)
# a.WhileBackWard(0.90, 0.001)
a.BestWhileBackWard(0.90)
a.printdata()
