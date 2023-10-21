from typing import Tuple
import torch
from torch import Tensor

# x1 = torch.rand((3, 3)).cuda() < 0.5
# x2 = torch.rand((3, 3)).cuda() < 0.5
# print(x1)
# print(x2)

# def get_tp_fp_fn_tn(x: Tensor, y: Tensor) -> Tuple[int, int, int, int]:
#     tp = torch.logical_and(x,y).sum()
#     fp = x.sum() - tp
#     fn = torch.logical_not(torch.logical_nand(x,y).sum())
#     tn = 0

#     return tp, fp, fn, tn
#     pass

# def dice_score(x: Tensor, y: Tensor) -> Tensor:
#     pass

# print(get_tp_fp_fn_tn(x1,x2))


#=====================================
from torch.utils.data import TensorDataset

x = torch.rand(3,2).cuda()
y = torch.rand(3,2).cuda()


def getEuclidianDistance(x: Tensor, y: Tensor) -> float:
    return torch.square(x - y).sum().sqrt()

print(x[0])
print(y[0])
print(getEuclidianDistance(x[0],y[0]))

def getEuclidianDistanceBatched(x: Tensor, y: Tensor) -> Tensor:
    return torch.square(x - y).sum().sqrt()
    pass

print(x)
print(y)
print (x-y)
print(torch.square(x - y).sum())
print(getEuclidianDistanceBatched(x,y))




#====================================

# %matplotlib inline
# import matplotlib.pyplot as plt
# import cv2 as cv

# img = torch.from_numpy(cv.imread("test_image.jpg"))
# plt.imshow(img)
# plt.show()

# # img_rgb = img[::[0,1,2]]  # your code here


# # plt.imshow(img)
# # plt.show()

# img_greyscale = img.float().mean(dim=1)

# plt.imshow(img_greyscale)
# plt.show()