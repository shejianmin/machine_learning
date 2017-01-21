#!/usr/bin/env python
# encoding: utf-8
# File Name: HiddenLayer.py
# Author: Jiezhong Qiu
# Create Time: 2014/11/22 13:46
# TODO: implemetn HiddenLayer

import numpy as np
import util

class InputLayer(object):
    def __init__(self, rng, n_in, n_out, activation, activationPrime,
        nxt, Lambda = 0., alpha = .001):
        W = np.asarray(
            rng.uniform(
                low = -np.sqrt(6. / (n_in + n_out)),
                high = np.sqrt(6. / (n_in + n_out)),
                size = (n_out, n_in)
            ),
            dtype = util.FLOAT
        )
        b = np.zeros(n_out, dtype=util.FLOAT).reshape(n_out, 1)
        self.W, self.b = W * 4, b
        self.n_in, self.n_out = n_in, n_out
        self.activation, self.activationPrime = activation, activationPrime
        self.nxt = nxt
        self.Lambda = Lambda #weight decay
        self.alpha = alpha #learning rate
    def process(self, x, bp = True):
        '''
            x : (n_in * m)
            W : (n_out * n_in)
        '''
        m = x.shape[1]
        z = np.dot(self.W, x) + np.tile(self.b, (1, m))
        a_out = self.activation(z)
        self.cost = self.Lambda / 2 * ( (self.W ** 2).sum() )
        self.nxt.process(a_out, bp)
        if bp == True:
            delta = self.nxt.getdelta()
            gradW = np.dot(delta, x.T) / m + self.Lambda * self.W
            #gradb = np.dot(delta, np.ones((m, 1))) / m
            gradb = np.sum(delta, axis=1).reshape(self.n_out, 1) / m
            self.W -= self.alpha * gradW
            self.b -= self.alpha * gradb
        #return np.dot( np.dot(a_in, 1 - a_in), np.dot(self.W.T, delta) )

class OutputLayer(object):
    def __init__(self, n_in, activation, activationPrime, errorFunc, costFunc):
        self.n_in = n_in
        self.activation, self.activationPrime = activation, activationPrime
        self.errorFunc, self.costFunc = errorFunc, costFunc
    def setstd(self, std):
        self.std = std

    def getdelta(self):
        return self.delta

    def getresult(self):
        return self.result

    def process(self, a_in, bp = True):
        self.result = a_in
        if bp:
            m = a_in.shape[1]
            #self.delta = (a_in - self.std) * self.activationPrime(a_in) if bp else None
            self.delta = self.errorFunc(self.std, a_in)
            #error = a_in - self.std
            #self.cost = .5 * (( error ** 2 ).sum()) / m
            self.cost = self.costFunc(self.std, a_in) / m
        #return ( a_in - self.std ) * ( a_in * (1 - a_in) )
        #a_in*(1-a_in) (n_in * m)
        #a_in - self.std n_in * m
class HiddenLayer(object):
    def __init__(self, rng, n_in, n_out, activation, activationPrime,
            nxt, Lambda = 0., alpha = .001):
        W = np.asarray(
            rng.uniform(
                low = -np.sqrt(6. / (n_in + n_out)),
                high = np.sqrt(6. / (n_in + n_out)),
                size = (n_out, n_in)
            ),
            dtype = util.FLOAT
        )
        b = np.zeros(n_out, dtype=util.FLOAT).reshape(n_out, 1)
        self.W, self.b = W * 4, b
        self.n_in, self.n_out = n_in, n_out
        self.activation, self.activationPrime = activation, activationPrime
        self.nxt = nxt
        self.Lambda = Lambda #weight decay
        self.alpha = alpha #learning rate

    def getdelta(self):
        return self.delta

    def update(self, a_in):
        m = a_in.shape[1]
        delta = self.nxt.getdelta()
        gradW = np.dot(delta, a_in.T) / m + self.Lambda * self.W
        #gradb = np.dot(delta, np.ones((m, 1))) / m
        gradb = np.sum(delta, axis=1).reshape(self.n_out, 1) / m
        self.W -= self.alpha * gradW
        self.b -= self.alpha * gradb
        self.delta = np.dot(self.W.T, delta) * self.activationPrime(a_in)

    def process(self, a_in, bp = True):
        m = a_in.shape[1]
        z = np.dot(self.W, a_in) + np.tile(self.b, (1, m))
        a_out = self.activation(z)
        self.cost = self.Lambda / 2 * ( (self.W ** 2).sum() )
        self.nxt.process(a_out , bp)
        if bp:
            self.update(a_in)

class SoftmaxLayer(HiddenLayer):
    def process(self, a_in, bp = True):
        m = a_in.shape[1]
        z = np.dot(self.W, a_in) + np.tile(self.b, (1, m))
        logsum = np.log(np.sum(np.exp(z), axis=0))
        a_out = np.exp(z - np.expand_dims(logsum, axis=0))
        self.cost = self.Lambda / 2 * ( (self.W ** 2).sum() )
        self.nxt.process(a_out , bp)
        if bp:
            self.update(a_in)
if __name__ == "__main__":
    pass


