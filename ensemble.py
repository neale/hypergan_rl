import torch
import torch.nn as nn
import torch.nn.functional as F
from hypergan_base import HyperGAN_Base

def swish(x):
    return x * torch.sigmoid(x)

nh = 16
ni = 32
no = 512

class GeneratorW1(nn.Module):
    def __init__(self, d_action, d_state):
        super(GeneratorW1, self).__init__()
        self.d_action = d_action
        self.d_state = d_state
        self.linear1 = nn.Linear(ni, nh, bias=False)
        self.linear2 = nn.Linear(nh, nh, bias=False)
        self.linear3 = nn.Linear(nh, no * (d_action+d_state) + no, bias=False) # [pop, 25, nh], [pop, 1, nh]

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        #x = F.leaky_relu(self.linear2(x))
        x = self.linear3(x)
        w, b = x[:, :no*(self.d_action+self.d_state)], x[:, -no:]
        w = w.view(-1, no, self.d_action+self.d_state)
        b = b.view(-1, 1, no)
        return (w, b)


class GeneratorW2(nn.Module):
    def __init__(self):
        super(GeneratorW2, self).__init__()
        self.linear1 = nn.Linear(ni, nh, bias=False)
        self.linear2 = nn.Linear(nh, nh, bias=False)
        self.linear3 = nn.Linear(nh, no*no + no, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        #x = F.leaky_relu(self.linear2(x))
        x = self.linear3(x)
        w, b = x[:, :no*no], x[:, -no:]
        w = w.view(-1, no, no)
        b = b.view(-1, 1, no)
        return (w, b)


class GeneratorW3(nn.Module):
    def __init__(self):
        super(GeneratorW3, self).__init__()
        self.linear1 = nn.Linear(ni, nh, bias=False)
        self.linear2 = nn.Linear(nh, nh, bias=False)
        self.linear3 = nn.Linear(nh, no*no + no, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        #x = F.leaky_relu(self.linear2(x))
        x = self.linear3(x)
        w, b = x[:, :no*no], x[:, -no:]
        w = w.view(-1, no, no)
        b = b.view(-1, 1, no)
        return (w, b)


class GeneratorW4(nn.Module):
    def __init__(self):
        super(GeneratorW4, self).__init__()
        self.linear1 = nn.Linear(ni, nh, bias=False)
        self.linear2 = nn.Linear(nh, nh, bias=False)
        self.linear3 = nn.Linear(nh, no*no + no, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        #x = F.leaky_relu(self.linear2(x))
        x = self.linear3(x)
        w, b = x[:, :no*no], x[:, -no:]
        w = w.view(-1, no, no)
        b = b.view(-1, 1, no)
        return (w, b)


class GeneratorW5(nn.Module):
    def __init__(self, d_state):
        super(GeneratorW5, self).__init__()
        self.d_state = d_state
        self.linear1 = nn.Linear(ni, nh, bias=False)
        self.linear2 = nn.Linear(nh, nh, bias=False)
        self.linear3 = nn.Linear(nh, no*d_state + d_state, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        #x = F.leaky_relu(self.linear2(x))
        x = self.linear3(x)
        w, b = x[:, :no*self.d_state], x[:, -self.d_state:]
        w = w.view(-1, self.d_state, no)
        b = b.view(-1, 1, self.d_state)
        return (w, b)

def orthogonal_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight)

def normal_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)


class HyperGAN(HyperGAN_Base):
    
    def __init__(self, device, d_action, d_state):
        super(HyperGAN, self).__init__(device, d_action, d_state)
        self.generator = self.Generator(device, d_action, d_state)

    class Generator(object):
        def __init__(self, device, d_action, d_state):
            self.W1 = GeneratorW1(d_action, d_state).to(device)
            self.W2 = GeneratorW2().to(device)
            self.W3 = GeneratorW3().to(device)
            self.W4 = GeneratorW4().to(device)
            self.W5 = GeneratorW5(d_state).to(device)
            
            for m in [self.W1, self.W2, self.W3, self.W4, self.W5]:
                m.apply(orthogonal_init)
                # m.apply(normal_init)

        def __call__(self, x):
            w1, b1 = self.W1(x[0])
            w2, b2 = self.W2(x[1])
            w3, b3 = self.W3(x[2])
            w4, b4 = self.W4(x[3])
            w5, b5 = self.W5(x[4])
            layers = [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5]
            return layers
        
        def get_all_params(self):
            W1_lst = list(self.W1.parameters())
            W2_lst = list(self.W2.parameters())
            W3_lst = list(self.W3.parameters())
            W4_lst = list(self.W4.parameters())
            W5_lst = list(self.W5.parameters())
            return W1_lst + W2_lst + W3_lst + W4_lst + W5_lst 

        def as_list(self):
            return [self.W1, self.W2, self.W3, self.W4, self.W5]
    
    def eval_all(self, Z, data):
        w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 = Z
        x = torch.baddbmm(b1, data, w1.transpose(1, 2))
        x = F.leaky_relu(x)
        x = torch.baddbmm(b2, x, w2.transpose(1, 2))
        x = F.leaky_relu(x)
        x = torch.baddbmm(b3, x, w3.transpose(1, 2))
        x = F.leaky_relu(x)
        x = torch.baddbmm(b4, x, w4.transpose(1, 2))
        x = F.leaky_relu(x)
        x = torch.baddbmm(b5, x, w5.transpose(1, 2))
        return x
 
    """ functional model for training """
    def eval_f(self, Z, data):
        w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 = Z
        x = F.linear(data, w1, bias=b1)
        x = F.leaky_relu(x)
        x = F.linear(x, w2, bias=b2)
        x = F.leaky_relu(x)
        x = F.linear(x, w3, bias=b3)
        x = F.leaky_relu(x)
        x = F.linear(x, w4, bias=b4)
        x = F.leaky_relu(x)
        x = F.linear(x, w5, bias=b5)
        return x
