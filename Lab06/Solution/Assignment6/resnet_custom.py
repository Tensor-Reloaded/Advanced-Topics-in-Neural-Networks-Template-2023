import torch

from Assignment6.model import Model


class ResnetCustom(Model):
    def __init__(self, model, **kwargs):
        super(ResnetCustom, self).__init__(**kwargs)

        for m in model.modules():
            m = m.to(self.device)
            m.requires_grad_(False)

        upsample1 = torch.nn.Upsample(scale_factor=(2, 2)).to(self.device)
        upsample2 = torch.nn.Upsample(scale_factor=(2, 2)).to(self.device)
        upsample3 = torch.nn.Upsample(scale_factor=(2, 2)).to(self.device)

        added_1 = torch.nn.Flatten().to(self.device)
        added_4 = torch.nn.BatchNorm1d(2048).to(self.device)
        added_4.weight.data.fill_(1)
        added_4.bias.data.zero_()
        added_2 = torch.nn.Linear(2048, 1024).to(self.device)
        torch.nn.init.xavier_normal_(added_2.weight)
        added_3 = torch.nn.ReLU().to(self.device)
        added_11 = torch.nn.Dropout(0.5)
        added_7 = torch.nn.BatchNorm1d(1024).to(self.device)
        added_7.weight.data.fill_(1)
        added_7.bias.data.zero_()
        added_5 = torch.nn.Linear(1024, 512).to(self.device)
        torch.nn.init.xavier_normal_(added_5.weight)
        added_6 = torch.nn.ReLU().to(self.device)
        added_12 = torch.nn.Dropout(0.5)
        added_10 = torch.nn.BatchNorm1d(512).to(self.device)
        added_10.weight.data.fill_(1)
        added_10.bias.data.zero_()
        added_8 = torch.nn.Linear(512, 10).to(self.device)
        torch.nn.init.xavier_normal_(added_8.weight)
        added_9 = torch.nn.Softmax().to(self.device)

        self.layers = torch.nn.Sequential(upsample1,
                                          upsample2,
                                          upsample3,
                                          *list(model.children())[:-1],
                                          added_1,
                                          added_4,
                                          added_2,
                                          added_3,
                                          added_11,
                                          added_7,
                                          added_5,
                                          added_6,
                                          added_12,
                                          added_10,
                                          added_8,
                                          added_9,
                                          )