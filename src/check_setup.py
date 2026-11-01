import torch, torchvision as tv
from src.utils import set_seed, get_device

set_seed(42)
device = get_device()
print("Enhet:", device)

tf = tv.transforms.Compose([tv.transforms.ToTensor(),
                            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test = tv.datasets.CIFAR10("data", train=False, download=True, transform=tf)
print("Testbilder:", len(test))

x, y = next(iter(torch.utils.data.DataLoader(test, batch_size=8)))
model = tv.models.resnet18(num_classes=10).to(device).eval()
with torch.inference_mode():
    out = model(x.to(device))
print("Indata:", tuple(x.shape), "-> utdata:", tuple(out.shape))
