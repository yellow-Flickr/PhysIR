import os

# PyTorch library
from torch.utils.data import DataLoader, DistributedSampler

try:
    from .datapipeline import *
    from .utils import *
except:
    from datapipeline import *
    from utils import *

def _load_paired_flat(root, low_sub, high_sub, verbose, label):
    """
    Load a flat paired dataset from root/low_sub and root/high_sub.
    Returns (list_low, list_high) or (None, None) if the path is missing.
    """
    low_dir  = os.path.join(root, low_sub)
    high_dir = os.path.join(root, high_sub)
    if not os.path.isdir(low_dir) or not os.path.isdir(high_dir):
        if verbose:
            print(f'  [{label}] skipped — path not found: {root}')
        return None, None
    low_paths  = sorted(os.path.join(low_dir,  p) for p in os.listdir(low_dir))
    high_paths = sorted(os.path.join(high_dir, p) for p in os.listdir(high_dir))
    if verbose:
        print(f'  [{label}] {len(low_paths)} pairs found')
    return low_paths, high_paths


def _load_paired_nested(root, low_sub, high_sub, verbose, label):
    """
    Load a nested paired dataset (one subfolder per video clip).
    Returns (list_low, list_high) or (None, None) if the path is missing.
    """
    low_dir  = os.path.join(root, low_sub)
    high_dir = os.path.join(root, high_sub)
    if not os.path.isdir(low_dir) or not os.path.isdir(high_dir):
        if verbose:
            print(f'  [{label}] skipped — path not found: {root}')
        return None, None
    clips_low  = [os.path.join(low_dir,  c) for c in os.listdir(low_dir)]
    clips_high = [os.path.join(high_dir, c) for c in os.listdir(high_dir)]
    low_paths  = flatten_list_comprehension(
        [sorted(os.path.join(c, f) for f in os.listdir(c)) for c in sorted(clips_low)])
    high_paths = flatten_list_comprehension(
        [sorted(os.path.join(c, f) for f in os.listdir(c)) for c in sorted(clips_high)])
    if verbose:
        print(f'  [{label}] {len(low_paths)} pairs found (nested clips)')
    return low_paths, high_paths


def _make_loader(low_paths, high_paths, batch_size, num_workers,
                 world_size, rank, tensor_transform):
    dataset = MyDataset_Crop(low_paths, high_paths, cropsize=None,
                             tensor_transform=tensor_transform, test=True)
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size,
                                     shuffle=True, rank=rank)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            drop_last=False, sampler=sampler)
    else:
        sampler = None
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True,
                             drop_last=False)
    return loader, sampler


def main_dataset_all_lol(rank=1, test_path='../../data/datasets/',
                         batch_size_test=1, verbose=False,
                         num_workers=1, world_size=1):
    """
    Loads whichever of {LOLv2-real, LOLv2-synth, LOLBlur} test splits are present.
    Missing datasets are skipped gracefully — no crash if LOLBlur/test/ is absent.
    Returns a dict keyed by dataset name; only present datasets are included.
    """
    if verbose:
        print('Loading All_LOL val sets from:', test_path)

    tensor_transform = transforms.ToTensor()
    test_loader = {}
    samplers    = []

    # ── LOLv2-real ────────────────────────────────────────────────────────────
    low, high = _load_paired_flat(
        os.path.join(test_path, 'LOL-v2/Real_captured/test'),
        'Low', 'Normal', verbose, 'lolv2-real')
    if low is not None:
        check_paths([low, high])
        loader, samp = _make_loader(low, high, batch_size_test, num_workers,
                                    world_size, rank, tensor_transform)
        test_loader['lolv2'] = {'loader': loader, 'adapter': False}
        if samp: samplers.append(samp)

    # ── LOLv2-synth ───────────────────────────────────────────────────────────
    low, high = _load_paired_flat(
        os.path.join(test_path, 'LOL-v2/Synthetic/test'),
        'Low', 'Normal', verbose, 'lolv2-synth')
    if low is not None:
        check_paths([low, high])
        loader, samp = _make_loader(low, high, batch_size_test, num_workers,
                                    world_size, rank, tensor_transform)
        test_loader['lolv2_synth'] = {'loader': loader, 'adapter': False}
        if samp: samplers.append(samp)

    # ── LOLBlur (nested clips) ─────────────────────────────────────────────────
    low, high = _load_paired_nested(
        os.path.join(test_path, 'LOLBlur/test'),
        'low_blur_noise', 'high_sharp_scaled', verbose, 'lolblur')
    if low is not None:
        check_paths([low, high])
        loader, samp = _make_loader(low, high, batch_size_test, num_workers,
                                    world_size, rank, tensor_transform)
        test_loader['lolblur'] = {'loader': loader, 'adapter': False}
        if samp: samplers.append(samp)

    if not test_loader:
        raise FileNotFoundError(
            f'No recognised val datasets found under {test_path}. '
            'Expected LOL-v2/ and/or LOLBlur/test/.')

    if verbose:
        print(f'  Active val sets: {list(test_loader.keys())}')

    return test_loader, (samplers if samplers else None)

if __name__ == '__main__':
    
    test_loader, samplers= main_dataset_all_lol(verbose = True, test_path='/mnt/valab-datasets/')

    