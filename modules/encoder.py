import torch
import torch.nn as nn

class Encoder(nn.Module):
  def __init__(self,input_size, output_size,batch_size=1):
    super(Encoder,self).__init__()
    self.conv1 = nn.Conv2d(in_channels=input_size,out_channels=8,kernel_size=3,stride=1,padding=1)
    self.conv2 = nn.Conv2d(in_channels= 8, out_channels=16, kernel_size=3, stride = 1, padding=1)
    self.conv3 = nn.Conv2d(in_channels= 16, out_channels=32, kernel_size=3, stride = 1, padding=1)
    self.activation_fn = torch.nn.ReLU()
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2,padding = 1)
    self.mapf = nn.Linear(32*2*4, output_size)

  def forward(self,x):
    x = self.activation_fn(self.conv1(x))
    x = self.pool(x)
    x = self.activation_fn(self.conv2(x))
    x = self.pool(x)
    x = self.activation_fn(self.conv3(x))
    x = self.pool(x)
    x = x.view(x.size(0),-1)
    return torch.nn.Sigmoid()(self.mapf(x))

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x



class Encoder1D(nn.Module):
  def __init__(self,input_size,output_size,batch_size=1):
    super(Encoder1D,self).__init__()
    self.conv1 = nn.Conv1d(in_channels=input_size,out_channels=8,kernel_size=5,stride=1,padding=1)
    self.conv2 = nn.Conv1d(in_channels= 8, out_channels=16, kernel_size=5, stride = 1, padding=1)
    self.conv3 = nn.Conv1d(in_channels= 16, out_channels=32, kernel_size=5, stride = 1, padding=1)
    self.activation_fn = torch.nn.ReLU()
    self.pool = nn.MaxPool1d(kernel_size=2, stride=2,padding = 1)
    self.mapf = nn.Linear(384, output_size)

  def forward(self,x):
    x = self.activation_fn(self.conv1(x))
    x = self.pool(x)
    x = self.activation_fn(self.conv2(x))
    x = self.pool(x)
    x = self.activation_fn(self.conv3(x))
    x = self.pool(x)
    x = x.view(x.size(0),-1)
    return torch.nn.Sigmoid()(self.mapf(x))