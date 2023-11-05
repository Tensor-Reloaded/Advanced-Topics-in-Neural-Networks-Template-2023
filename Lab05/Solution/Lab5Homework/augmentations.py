from torchvision.transforms import v2


class CutOrMixUp:
    def __init__(self):
        cut_mix = v2.CutMix(num_classes=10)
        mix_up = v2.MixUp(num_classes=10)
        self.cutmix_or_mixup_choice = v2.RandomChoice([cut_mix, mix_up])

    def __call__(self, data, labels):
        return self.cutmix_or_mixup_choice(data, labels)
