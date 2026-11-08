import torchvision as tv
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch

CIFAR_MEAN = (0.49139968, 0.48215841, 0.44653091)
CIFAR_STD = (0.24703223, 0.24348513, 0.26158784)


def get_transforms(train: bool):
    """Transformer for training (with augmentaiton) or validation (without)."""
    t = []
    if train:
        t += [
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
        ]
    t += [
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]
    return tv.transforms.Compose(t)



def get_loaders(batch_size: int = 128, num_workers: int = 0):
    train_aug = tv.datasets.CIFAR10(root="data", train=True, download=True,
                                    transform=get_transforms(train=True))
    train_eval = tv.datasets.CIFAR10(root="data", train=True, download=True,
                                     transform=get_transforms(train=False))
    test_data = tv.datasets.CIFAR10(root="data", train=False, download=True,
                                    transform=get_transforms(train=False))

    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(train_aug), generator=g).tolist()
    train_idx, val_idx = perm[:45_000], perm[45_000:]

    train_set = Subset(train_aug, train_idx)   # with aug
    val_set = Subset(train_eval, val_idx)      # without aug

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
    