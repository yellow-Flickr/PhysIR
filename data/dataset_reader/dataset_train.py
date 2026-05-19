import os
from pathlib import Path
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

try:
    from .datapipeline import MyDataset_Crop
except ImportError:
    from datapipeline import MyDataset_Crop

_SUPPORTED = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}


def _collect_images(directory):
    # rglob handles both flat datasets (LOLv2) and nested ones (LOLBlur video clips)
    return sorted(str(p) for p in Path(directory).rglob('*') if p.suffix in _SUPPORTED)


def create_train_loader(low_dir, high_dir,
                        crop_size=256,
                        batch_size=4,
                        num_workers=4,
                        use_flips=True,
                        pin_memory=True):
    '''
    Generic paired (low / high) training loader.

    low_dir  : directory of low-light images
    high_dir : directory of corresponding ground-truth images
    Files are sorted and matched by position — names must correspond 1-to-1.
    '''
    low_paths  = _collect_images(low_dir)
    high_paths = _collect_images(high_dir)

    if len(low_paths) == 0:
        raise FileNotFoundError(f'No images found in {low_dir}')
    if len(low_paths) != len(high_paths):
        raise ValueError(f'Mismatched image counts: {len(low_paths)} low, '
                         f'{len(high_paths)} high in {low_dir} / {high_dir}')

    flips = transforms.RandomHorizontalFlip(p=0.5) if use_flips else None
    dataset = MyDataset_Crop(
        images_low=low_paths,
        images_high=high_paths,
        cropsize=crop_size,
        tensor_transform=transforms.ToTensor(),
        flips=flips,
        test=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    return loader, dataset
