"""
This model predicts sin(theta+k) given the unit circle coorinates of theta
"""
import os
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam
from torch.nn import MSELoss
import math
import torch.nn as nn
import numpy as np
import pandas as pd

class AutoRec(Module):
    def __init__(self):
        super(AutoRec,self).__init__()
        self.hidden = nn.Parameter(torch.empty(2).normal_())
        
        
    def forward(self, data):
        out = torch.sum(self.hidden*data,axis=1)
        return out
        
    # train the model
    def fit(self,loader,lr=.01,epochs=100):
        # define the optimization
        optimizer = Adam(self.parameters(), lr=lr)
        # enumerate epochs
        for epoch in range(epochs):
            predictions,actuals = torch.empty(0),torch.empty(0)
            for data,actual in iter(loader):
                optimizer.zero_grad(set_to_none=True)
                yhat = self.forward(data).reshape((-1,))
                predictions = torch.hstack((predictions,yhat))
                actuals = torch.hstack((actuals,actual))
                loss = MSELoss()(yhat,actual)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                loss2 = MSELoss()(predictions,actuals)
                print(f"Epoch {epoch}/{epochs}; Current Train MSE: {loss2}")
    
    def evaluate(self,loader):
        self.training = False
        with torch.no_grad():
            predictions,actuals = torch.empty(0),torch.empty(0)
            for data,actual in iter(loader):
                yhat = self.forward(data).reshape((-1,))
                predictions = torch.hstack((predictions,yhat))
                actuals = torch.hstack((actuals,actual))
            with torch.no_grad():
                loss2 = MSELoss()(predictions,actuals)
                print(f"Test MSE: {loss2}")
class Loader:
    def __init__(self,batch_size):
        self.batch_size = batch_size
    def __len__(self):
           return 4*self.batch_size
    def __iter__(self):
        self.current_index = 0
        return self
    def __next__(self): # Python 2: def next(self)
        if self.current_index < self.__len__()-self.batch_size:
            output = torch.rand(self.batch_size)*2*math.pi
            y = torch.sin(output+.5*math.pi)
            x = torch.t(torch.vstack((torch.cos(output),torch.sin(output))))
            # print(x,y)
            self.current_index += self.batch_size
            return (x,y)
        else:
            raise StopIteration
                         
if __name__ == "__main__":

    model = AutoRec()
    print("Fitting model")
    loader = Loader(100)
    print(loader)
    model.fit(loader)
    print("Learned parameters",model.hidden.data)
    model.evaluate(loader)