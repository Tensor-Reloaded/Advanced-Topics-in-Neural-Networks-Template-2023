import torch
from torchvision.datasets import CIFAR10

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
        # For multi-gpu workstations, PyTorch will use the first available GPU (cuda:0), unless specified otherwise
        # (cuda:1).
    if torch.backends.mps.is_available():
        return torch.device('mos')
    return torch.device('cpu')

class CustomConv2D(torch.nn.Module):
    def __init__(self, weights):
        super(CustomConv2D, self).__init__()
        self.weights = weights
    
    def forward(self, data):
        batch_size, channels, height, width = data.shape
        ret_channels, _, k_height, k_width = self.weights.shape
        ret_height = height - k_height + 1
        ret_width = width - k_width + 1
        
        data_reshape = data.unfold(2, k_height, 1).unfold(3, k_width, 1)
        data_reshape = data_reshape.permute(0,2,3,1,4,5).contiguous().view(batch_size, ret_height, ret_width, -1)

        weights_reshape = self.weights.view(ret_channels, -1).t()

        results = torch.matmul(data_reshape, weights_reshape).view(batch_size, ret_height, ret_width, ret_channels)
        results = results.permute(0,3,1,2).contiguous()

        return results
    

def main():
    data = torch.randn(1,3,10,12)
    weights = torch.randn(2,3,4,5)

    CustomConv2DLayer = CustomConv2D(weights=weights)
    result = CustomConv2DLayer(data)

    print((torch.nn.functional.conv2d(data,weights) - result).abs().max())
    #print(result)
    #print(torch.nn.functional.conv2d(data,weights))

if __name__ == "__main__":
    main()