import torch

#  can also import arrays from numpy, not only from python
x = torch.rand((4, 8))
print(x.size(0))  # same as shape[0]

# first dimension in tensor is always batch dimension -> inmultirea opereaza direct peste
# elementele 2x2 (dim2xdim3x... trebuie sa se potriveasca)

# to('cuda'): datele se transfera de pe cpu (RAM) pe gpu
# diferenta intre view si reshape:
# -1 inseamna completarea restului de dimensiune: view nu recreeaza in memorie
# unsqueeze -- creaza o dimensiune goala in plus, pt usurinta calculelor
# stack (ex.: filtru) creeaza o dimensiune noua, cat continua pe o dimensiune deja existenta

# clip poate forta gradientii sa nu faca salturi prea mari --> limitare zgomot, outliers
# (in timp ce learning rate schimba toti gradientii deodata)


b = torch.rand((4, 8))
b[b < 0.7] = 0  # facem deja elementele 0, fiind deja considerate foarte aproape de 0, pemtru a nu pastra astfel noise



