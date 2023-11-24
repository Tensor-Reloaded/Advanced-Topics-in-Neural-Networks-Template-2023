import os
import torch
from torchvision.datasets import CIFAR10

class Handmade_conv2d_implementation():
    def __init__(self,weights):
        self.weights = weights
        
    def __call__(self,inp):
        #result=torch.zeros((inp.shape[0],w.shape[1],inp.shape[2]-w.shape[2]+1,inp.shape[3]-w.shape[3]+1))
        w=self.weights
        results=[]
        for img in inp:
            r_part=torch.zeros((w.shape[0],inp.shape[2]-w.shape[2]+1,inp.shape[3]-w.shape[3]+1))
            for idx in range(0,w.shape[0]):
                for i in range(0,inp.shape[2]-w.shape[2]+1):
                    for j in range(0,inp.shape[3]-w.shape[3]+1):
                        for ch in range(0,img.shape[0]):
                            p1=img[ch][i:i+w.shape[2],j:j+w.shape[3]]
                            p2=w[idx][ch]
                            r_part[idx][i][j]+=(img[ch][i:i+w.shape[2],j:j+w.shape[3]]*w[idx][ch]).sum().item()
            results.append(r_part)
        return torch.cat(results,dim=0)
                
        

def main():
    inp = torch.randn(1, 3, 10, 12)
    w = torch.rand(2, 3, 4, 5)
    custom_conv2d_layer=Handmade_conv2d_implementation(weights = w)
    out = custom_conv2d_layer(inp)
    #print(out)
    print((torch.nn.functional.conv2d(inp,w)-out).abs().max())
    #print(torch.nn.functional.conv2d(inp,w))

if __name__ == '__main__':
    main()
