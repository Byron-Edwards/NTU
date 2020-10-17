import torch
import torch.utils.data as data

from PIL import Image
import os
import os.path
from os import listdir
from os.path import isfile, join,isdir


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CelebA(data.Dataset):
    def __init__(self, root, ann_file, transform=None, target_transform=None, loader=default_loader, test=False):
        images = []
        targets = []
        if test:
            for d in [join(ann_file, d) for d in listdir(ann_file) if isdir(join(ann_file, d))]:
                images += [join(d, f) for f in listdir(d)]
            self.images = images
        else:
            for line in open(os.path.join(root, ann_file), 'r'):
                sample = line.split()
                if len(sample) != 41:
                    raise(RuntimeError("# Annotated face attributes of CelebA dataset should not be different from 40"))
                images.append(sample[0])
                targets.append([1 if int(i) == 1 else 0 for i in sample[1:] ])
            self.images = [os.path.join(root, 'img_align_celeba_png', img) for img in images]
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.test = test

    def __getitem__(self, index):
        path = self.images[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.test:
            return sample, path.split("/")[-1]

        target = self.targets[index]
        target = torch.LongTensor(target)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.images)
