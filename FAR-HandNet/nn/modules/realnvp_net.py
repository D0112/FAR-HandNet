import torch.nn as nn
import torch

class Real_nets(nn.Module):
    def __init__(self,device='cuda:0', c1=2, c2=64):
        super().__init__()
        # self.nets_l1 = nn.Linear(c1, c2,device=device,dtype=torch.float32)
        # self.nets_l2 = nn.Linear(c2, c2,device=device,dtype=torch.float32)
        # self.nets_l3 = nn.Linear(c2, c1, device=device,dtype=torch.float32)
        # self.nets_relu1 = nn.LeakyReLU()
        # self.nets_relu2 = nn.LeakyReLU()
        # self.nets_Tanh = nn.Tanh()
        # layers = nn.Sequential(nn.Linear(c1, c2,device=device,dtype=torch.float32),nn.LeakyReLU(),
        #                        nn.Linear(c2, c2,device=device,dtype=torch.float32),nn.LeakyReLU(),
        #                        nn.Linear(c2, c1, device=device,dtype=torch.float32),nn.Tanh())

        layers = nn.Sequential(nn.Linear(c1, c2, device=device, dtype=torch.float32), nn.BatchNorm1d(c2), nn.LeakyReLU(),
                               nn.Linear(c2, c2, device=device, dtype=torch.float32), nn.BatchNorm1d(c2), nn.LeakyReLU(),
                               nn.Linear(c2, c1, device=device, dtype=torch.float32), nn.BatchNorm1d(c1), nn.Tanh())

        self.block = torch.nn.ModuleList([layers for _ in range(6)])
    def __getitem__(self, index):
        return self.block[index]

    def forward(self, x):
        # return nn.Sequential(self.nets_l1, self.nets_relu1, self.nets_l2, self.nets_relu2, self.nets_l3, self.nets_Tanh)
        # return self.nets_Tanh(self.nets_l3(self.nets_relu2(self.nets_l2(self.nets_relu1(self.nets_l1(x))))))
        return self.block(x)




class Real_nett(nn.Module):
    def __init__(self, device='cuda:0', c1=2, c2=64):
        super().__init__()
        # self.nets_l1 = nn.Linear(c1, c2,device=device,dtype=torch.float32)
        # self.nets_l2 = nn.Linear(c2, c2,device=device,dtype=torch.float32)
        # self.nets_l3 = nn.Linear(c2, c1, device=device,dtype=torch.float32)
        # self.nets_relu1 = nn.LeakyReLU()
        # self.nets_relu2 = nn.LeakyReLU()
        # self.nets_Tanh = nn.Tanh()
        layers = nn.Sequential(nn.Linear(c1, c2, device=device, dtype=torch.float32), nn.BatchNorm1d(c2), nn.LeakyReLU(),
                               nn.Linear(c2, c2, device=device, dtype=torch.float32), nn.BatchNorm1d(c2), nn.LeakyReLU(),
                               nn.Linear(c2, c1, device=device, dtype=torch.float32), nn.BatchNorm1d(c1))
        self.block = torch.nn.ModuleList([layers for _ in range(6)])
    def __getitem__(self, index):
        return self.block[index]

    def forward(self, x):
        # return nn.Sequential(self.nets_l1, self.nets_relu1, self.nets_l2, self.nets_relu2, self.nets_l3, self.nets_Tanh)
        # return self.nets_Tanh(self.nets_l3(self.nets_relu2(self.nets_l2(self.nets_relu1(self.nets_l1(x))))))
        return self.block(x)