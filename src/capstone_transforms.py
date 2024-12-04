import torch

from torchvision.transforms import v2

"""
Normalizing the vectors is done according to the documentation for pytorch.
The original ImageNet images were normalized in that way.
"""
# Transformation without data augmentation
def get_transform(size=1024):
    transforms = v2.Compose([v2.ToImage(),
                            v2.Resize((size, size), antialias=True), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    return transforms


# Transformations incl. data augmention
def get_da_transform(size=1024):
    transforms = v2.Compose([v2.ToImage(),
                            v2.Resize((size, size), antialias=True),
                            v2.RandomHorizontalFlip(p=0.5),
                            v2.RandomVerticalFlip(p=0.5),
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    return transforms
