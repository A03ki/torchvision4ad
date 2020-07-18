# torchvision for Anomaly Detection

You can use the MVTec Anomaly Detection Dataset.

## Installation

pip:

```bash
$ pip install torchvision4ad
```

From source:

```bash
$ python setup.py install
```

## Usage

You can use one of the MVTec AD Dataset names {'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'}.

```python
from torchvision4ad.datasets import MVTecAD


root = "mvtec_ad"
dataset_name = "bottle"
mvtec_ad = MVTecAD(root, dataset_name, train=True, download=True)
for (img, target) in mvtec_ad:
    ...
```

Of course, you can also give a function/transform takes in an PIL image and returns a transformed version.

```python
import torchvision.transforms as transforms

from torchvision4ad.datasets import MVTecAD


transform = transforms.Compose([transforms.Resize([64, 64]),
                                transforms.ToTensor()])
mvtec_ad = MVTecAD("mvtec_ad", "bottle", train=True, transform=transform, download=True)
```
