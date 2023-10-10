from typing import Union
from torch import Tensor
import torch


def get_normal_tensors(x: Tensor) -> Union[Tensor, None]:
    print("x: ", x, sep="\t\t")
    norms_batch = x.norm(dim=(1, 2))
    print("Norms (per gradient): ", norms_batch, sep="\t\t")
    miu = norms_batch.mean()
    print("Mean (of norms per gradient): ", miu, sep="\t\t")
    st_dev = norms_batch.std()  # correction = 1
    print("Standard deviation: ", st_dev, sep="\t\t")
    mask1 = x.norm(dim=(1, 2)) <= (miu + 1.5 * st_dev)
    x_1 = x[mask1]
    mask2 = x_1.norm(dim=(1, 2)) >= (miu - 1.5 * st_dev)
    return x_1[mask2] if len(x_1[mask2]) > 0 else None
    # mask = x.norm(dim=(1, 2)) - miu <= abs(st_dev * 1.5)
    # return x[mask]
    # se puteau si inmulti mask1 si mask2


print(get_normal_tensors(torch.rand((4, 2, 3))))  # smaller example to work on
# print(len(get_normal_tensors(torch.rand((100, 10, 256)))))  # B = 100, N = 10, M = 256
# elementele sunt de forma N, M (pe cazul din enunt, (10, 256))
