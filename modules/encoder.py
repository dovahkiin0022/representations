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
        self.mapf = nn.Linear(32*2*4 + 1, output_size)

    def hidden_rep(self,x):
        x = self.activation_fn(self.conv1(x))
        x = self.pool(x)
        x = self.activation_fn(self.conv2(x))
        x = self.pool(x)
        x = self.activation_fn(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0),-1)
        return x


    def forward(self,x,x1):
        x = self.hidden_rep(x)
        xf = torch.cat((x,x1),axis=1)
        return torch.nn.Sigmoid()(self.mapf(xf))

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x



class Encoder1D(nn.Module):
    def __init__(self,input_size,output_size,batch_size=1):
      super(Encoder1D,self).__init__()
      self.conv1 = nn.Conv1d(in_channels=input_size,out_channels=8,kernel_size=9,stride=1,padding=1)
      self.conv2 = nn.Conv1d(in_channels= 8, out_channels=16, kernel_size=9, stride = 1, padding=1)
      self.conv3 = nn.Conv1d(in_channels= 16, out_channels=32, kernel_size=9, stride = 1, padding=1)
      self.activation_fn = torch.nn.ReLU()
      self.pool = nn.MaxPool1d(kernel_size=2, stride=2,padding = 1)
      self.mapf = nn.Linear(288+1, output_size)

    def hidden_rep(self,x):
        x = self.activation_fn(self.conv1(x))
        x = self.pool(x)
        x = self.activation_fn(self.conv2(x))
        x = self.pool(x)
        x = self.activation_fn(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0),-1)
        return x


    def forward(self,x,x1):
        x = self.hidden_rep(x)
        xf = torch.cat((x,x1),axis=1)
        return torch.nn.Sigmoid()(self.mapf(xf))

class EncoderDNN(nn.Module):
    def __init__(self,in_size,  n_hidden_layers, hidden_size, out_size):
        super(EncoderDNN,self).__init__()
        def block(in_feat,out_feat):
            layers = [nn.Linear(in_feat,out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
    
        self.hidden_rep = nn.Sequential(*block(in_size,hidden_size),
        *n_hidden_layers*block(hidden_size,hidden_size))
        self.mapf = nn.Linear(hidden_size+1,out_size)
    
    def forward(self,x,x1):
        x = self.hidden_rep(x)
        xf = torch.cat((x,x1),axis=1)
        return torch.nn.Sigmoid()(self.mapf(xf))

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)