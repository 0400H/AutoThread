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
    def __init__(self, batch, dataset, param) :
        self.batch = batch
        self.param = param
        self.label = dataset[0]
        self.case = dataset[1]
        self.case_len = len(self.case)
        self.label_len = len(self.label)
        self.param_len = len(self.case[0])
        self.accuracy = 0
        self.loss = 0
        self.loss_mean = 0
        self.learn_rate = 0.0001

        if (self.case_len != self.label_len) :
            print('The length of case and label is not match!')
            return -1
        elif (self.batch > self.case_len) :
            print('Batch is longer than the length of case and label!')
            return -1
        else :
            self.input = t.ones(self.batch, self.param_len, requires_grad = False)
            self.weight = t.ones(self.param_len, 1, requires_grad = True)
            self.right = t.ones(self.batch, 1, requires_grad = False)
            self.output = t.ones(self.batch, 1, requires_grad = True)

    def printdata(self) :
        self.__batch()
        self.__forward()
        self.__loss_mean().backward()
        self.__upgrade_weight()
        print('input:', self.input.data)
        print('weight:', self.weight.data)
        print('weight.grad:', self.weight.grad)
        print('loss_rate:' % self.loss_rate)
        print('accuracy:%.2f' % self.accuracy)
        for i in range(0, self.batch) :
            print('number:', i, 'output:%.1f' % self.output.data[i].item(), 'right:%.1f' % self.right.data[i].item())

    def __batch(self) :
        for i in range(0, self.batch) :
            random_id = randint(0, self.case_len - 1)
            self.input[i] = T(self.case[random_id])
            self.right[i] = T(self.label[random_id])

    def __forward(self) :
        self.__batch()
        for j in range(0, self.batch) :
            self.output = F.relu(self.input.mm(self.weight))
            # self.output = self.input.mm(self.weight)

    def __loss(self) :
        self.loss = F.pairwise_distance(self.output, self.right, 2, keepdim = True)

    def __loss_mean(self) :
        self.__loss()
        self.loss_mean = self.loss.mean()
        return self.loss_mean

    def Export(self, source_in, source_out) :
        t.save(source_in, source_out)

    def Import(self, source_in, source_out) :
        source_out = t.load(source_in)

    def __upgrade_weight(self) :
        with t.no_grad() :
            self.weight -= self.weight.grad * self.learn_rate
            self.Export(self.weight, self.param)

    def __zero_grad(self) :
        with t.no_grad() :
            self.weight.grad.zero_()

    def __accuracy(self) :
        self.acycurac = 0
        self.loss_rate = (self.output - self.right).abs_() / self.right
        for i in range(0, self.batch) :
            if (self.loss_rate[i][0].item() <= 0.05) :
                self.accuracy += 1
        self.accuracy = self.accuracy / self.batch

    def BackWard(self, learn_rate, iteration) :
        self.iteration = iteration
        self.learn_rate = learn_rate
        for i in range(0, self.iteration) :
            self.__batch()
            self.__forward()
            self.__accuracy()
            self.__loss_mean().backward()
            self.__upgrade_weight()
            self.__zero_grad()
            if ((iteration != 1) and (int(i + 1) % int(100) == 0)) :
                print('iteration:', i + 1, 'loss_mean:%.4f' % self.__loss_mean().item(), 'accuracy:%.4f' % self.accuracy)

    def WhileBackWard(self, accuracy, learn_rate) :
        key = 0
        loop = 0
        while (key < 500) :
            self.BackWard(learn_rate, 1)
            if (int(loop + 1) % int(100) == 0) :
                print('iteration:', loop + 1, 'loss_mean:%.4f' % self.__loss_mean().item(), 'accuracy:%.4f' % self.accuracy)
            if (self.accuracy >= accuracy) :
                key += 1
            else:
                key = 0
            loop += 1

    def BestWhileBackWard(self, accuracy) :
        key = 0
        learn_rate = 0
        loop = 0
        sum = 0
        while (key <= accuracy) :
            if (self.accuracy > 0.1) :
                learn_rate = 0.0005
            elif (self.accuracy > 0.2) :
                learn_rate = 0.00001
            elif (self.accuracy > 0.3) :
                learn_rate = 0.000005
            elif (self.accuracy > 0.4) :
                learn_rate = 0.0000001
            elif (self.accuracy > 0.5) :
                learn_rate = 0.00000005
            elif (self.accuracy > 0.6) :
                learn_rate = 0.00000001
            elif (self.accuracy > 0.7) :
                learn_rate = 0.000000005
            elif (self.accuracy > 0.8) :
                learn_rate = 0.000000001
            elif (self.accuracy > 0.9) :
                learn_rate = 0.0000000005
            else :
                learn_rate = 0.001

            self.BackWard(learn_rate, 1)

            if (int(loop + 1) % int(100) == 0) :
                print('loss_mean:%.4f' % self.__loss_mean().item(), 'accuracy:%.4f' % self.accuracy)

            if (loop == 999) :
                key = sum / 1000
                sum = 0
                loop = 0
            else :
                sum += self.accuracy
                loop += 1
                key = 0


if __name__ == '__main__':
    dataset = param2vector('./testcase.param')
    a = MultipleRegression(20, dataset, './param.bin')
    a.Import('./param.bin', a.weight)
    # a.BackWard(0.00000001, 10000)
    # a.WhileBackWard(0.90, 0.001)
    a.BestWhileBackWard(0.55)
    a.printdata()
