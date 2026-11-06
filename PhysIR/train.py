"""
DarkIR — single-device and multi-GPU (DDP) training with physics-guided losses.

Single GPU:
    python train.py -p options/train/pilot_baseline.yml

Multi-GPU (e.g. 2× T4 on Kaggle):
    torchrun --nproc_per_node=2 train.py -p options/train/pilot_baseline.yml
    # or equivalently:
    python -m torch.distributed.run --nproc_per_node=2 train.py -p ...

DDP behaviour:
  - Training data is sharded across ranks via DistributedSampler.
  - Per-epoch losses are all_reduced so logged values are true dataset means.
  - Eval, logging, and checkpointing run on rank 0 only; rank 1 waits at a
    barrier. Val loader is NOT sharded — rank 0 evaluates the full val set,
    which is correct because DDP keeps weights identical across ranks at all times.
  - LPIPS and pyiqa models are instantiated on rank 0 only.
"""

import os
import copy
import math
import time
import random
import argparse
import datetime
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import numpy as np
from tqdm import tqdm
from pathlib import Path

from options.options import parse
from archs.DarkIR import DarkIR
from archs import create_optim_scheduler, save_checkpoint
from data import create_train_data, create_test_data
from data.dataset_reader.dataset_train import create_train_loader, _collect_images
from losses import create_loss, create_physics_loss, calculate_loss, SSIM
from utils.metrics_logger import MetricsLogger

try:
    from lpips import LPIPS as _LPIPS
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False

try:
    import pyiqa
    _PYIQA_AVAILABLE = True
except ImportError:
    _PYIQA_AVAILABLE = False

import torchvision.transforms as transforms

try:
    from data.dataset_reader.datapipeline import MyDataset_Crop
except ImportError:
    from datapipeline import MyDataset_Crop

# ── DDP helpers ───────────────────────────────────────────────────────────────

def setup_ddp(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')
    dist.init_process_group('nccl', rank=rank, world_size=world_size, device_id=torch.device(f'cuda:{rank}'), timeout=datetime.timedelta(hours=2))
    torch.cuda.set_device(rank)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_ddp():
    return dist.is_available() and dist.is_initialized()


def reduce_scalar(value, device):
    """Sum a Python float across all ranks, return mean."""
    t = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() / dist.get_world_size()


# ── Model helpers ─────────────────────────────────────────────────────────────

def build_model(opt_network):
    return DarkIR(
        img_channel        = opt_network['img_channels'],
        width              = opt_network['width'],
        middle_blk_num_enc = opt_network['middle_blk_num_enc'],
        middle_blk_num_dec = opt_network['middle_blk_num_dec'],
        enc_blk_nums       = opt_network['enc_blk_nums'],
        dec_blk_nums       = opt_network['dec_blk_nums'],
        dilations          = opt_network['dilations'],
        extra_depth_wise   = opt_network['extra_depth_wise'],
    )


def load_weights(model, path):
    """Load pretrained or resumed checkpoint; strips DDP module. prefix."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    # Resume checkpoint written by this trainer has both 'epoch' and 'model'.
    # Check this before bare 'params' because our save writes both keys.
    if 'epoch' in ckpt and 'model' in ckpt:
        state = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
        model.load_state_dict(state, strict=False)
        print(f'Resumed from {path} (epoch {ckpt["epoch"]})')
        return model, ckpt.get('optimizer'), ckpt.get('scheduler'), ckpt['epoch']
    if 'model_state_dict' in ckpt:
        state = {k.replace('module.', ''): v
                 for k, v in ckpt['model_state_dict'].items()}
        model.load_state_dict(state, strict=False)
        print(f'Resumed from {path} (epoch {ckpt.get("epoch", "?")})')
        return model, ckpt.get('optimizer_state_dict'), \
               ckpt.get('scheduler_state_dict'), ckpt.get('epoch', 0)
    if 'params' in ckpt:
        # Bare pretrained weights only — no epoch info
        model.load_state_dict(ckpt['params'])
        print(f'Loaded pretrained weights from {path}')
        return model, None, None, 0
    raise ValueError(f'Unrecognised checkpoint format in {path}')


def restore_phys_state(phys_losses, path, device, rank=0):
    """
    Restore learnable physics-loss parameters (e.g. blur PSF log_sigma) on resume.
    These live in the loss modules — not the model — so load_weights does not cover
    them. Without this, a resumed run silently reinitializes the PSF to its init value
    (observed as a σ discontinuity at the resume epoch).
    """
    if not phys_losses:
        return
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        return
    phys_state = ckpt.get('phys_state')
    if not phys_state:
        if rank == 0:
            print('  No phys_state in checkpoint — physics-loss params left at init '
                  '(expected for checkpoints written before this fix).')
        return
    for name, module in phys_losses.items():
        if name in phys_state:
            module.load_state_dict(phys_state[name], strict=False)
            module.to(device)
    if rank == 0 and 'blur_loss' in phys_losses and hasattr(phys_losses['blur_loss'], 'log_sigma'):
        s = phys_losses['blur_loss'].log_sigma.exp().item()
        print(f'  Restored physics-loss state (blur σ={s:.3f})')


class ModelEMA:
    """
    Exponential moving average of model weights. The EMA copy is what we EVALUATE and
    REPORT (smoother, usually a small quality gain); the online weights keep training.

    Decay warms up so the early average isn't anchored to the random init:
        d(t) = decay * (1 - exp(-updates / warmup))
    A copy of params AND buffers is kept; non-float buffers are copied verbatim.
    Rank-0 only (eval/save happen there). Cheap for a 3.3M-param model (~13 MB).
    """

    def __init__(self, model, decay=0.9999, warmup=2000):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay   = decay
        self.warmup  = max(1, warmup)
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / self.warmup))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach().to(v.device), alpha=1.0 - d)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd)


# ── Eval helpers ──────────────────────────────────────────────────────────────

_ssim_metric = SSIM(data_range=1.)


def eval_metrics(model, loader, device, lpips_fn):
    """PSNR / SSIM / LPIPS on the full loader. Run on rank 0 only."""
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    psnrs, ssims, lpips_vals = [], [], []
    with torch.no_grad():
        for high, low in loader:
            high, low = high.to(device), low.to(device)
            pred = raw_model(low, side_loss=False).clamp(0, 1)
            _, _, H, W = high.shape
            pred = pred[:, :, :H, :W]
            mse = torch.mean((pred - high) ** 2).item()
            psnrs.append(20 * np.log10(1.0 / (np.sqrt(mse) + 1e-8)))
            _ssim_metric.to(device)
            ssims.append(_ssim_metric(pred, high).item())
            lp = lpips_fn(pred * 2 - 1, high * 2 - 1)
            lpips_vals.append(lp.mean().item())
    raw_model.train()
    return {
        'psnr':  float(np.mean(psnrs)),
        'ssim':  float(np.mean(ssims)),
        'lpips': float(np.mean(lpips_vals)),
    }


def eval_perceptual(model, loader, device, metrics_dict):
    """NIQE / NRQM / MUSIQ (no-reference). Run on rank 0 only."""
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    scores = {k: [] for k in metrics_dict}
    with torch.no_grad():
        for _, low in loader:
            low  = low.to(device)
            pred = raw_model(low, side_loss=False).clamp(0, 1)
            for name, metric in metrics_dict.items():
                scores[name].append(metric(pred).mean().item())
    raw_model.train()
    return {k: float(np.mean(v)) for k, v in scores.items()}


def save_visual_samples(model, loader, device, out_dir, epoch, num_samples=4):
    """
    Save qualitative comparison grids: each row is [input | prediction | ground-truth].
    Deterministic — always uses the first `num_samples` val items so the same scenes
    are tracked across epochs. Rank 0 only. Returns the saved PNG path.
    """
    import torchvision.utils as vutils
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    collected = 0
    with torch.no_grad():
        for high, low in loader:
            high, low = high.to(device), low.to(device)
            pred = raw_model(low, side_loss=False).clamp(0, 1)
            _, _, H, W = high.shape
            pred = pred[:, :, :H, :W]
            for b in range(high.size(0)):
                rows.extend([low[b, :, :H, :W].cpu(), pred[b].cpu(), high[b].cpu()])
                collected += 1
                if collected >= num_samples:
                    break
            if collected >= num_samples:
                break

    raw_model.train()
    if not rows:
        return None
    grid = vutils.make_grid(rows, nrow=3, padding=4, normalize=False)
    out_path = Path(out_dir) / f'samples_epoch_{epoch:03d}.png'
    vutils.save_image(grid, str(out_path))
    return str(out_path)


def get_phase(epoch, cfg):
    if epoch >= cfg['phase4_start']: return 4
    if epoch >= cfg['phase3_start']: return 3
    if epoch >= cfg['phase2_start']: return 2
    return 1


_PHASE_LABELS = {
    1: 'L1 + Edge',
    2: 'L1 + Edge + Retinex',
    3: 'L1 + Edge + Retinex + BlurGrad',
    4: 'L1 + Edge + Retinex + BlurGrad + PhaseFreq  [FULL]',
}

# ── Core training worker (one rank) ──────────────────────────────────────────

def train_worker(rank, world_size, config_path):
    ddp = world_size > 1
    if ddp:
        setup_ddp(rank, world_size)

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available()
                          else ('mps' if torch.backends.mps.is_available() else 'cpu'))

    opt      = parse(config_path)
    opt_save = opt['save']
    opt_train = opt['train']

    # Optional deterministic seeding (per-rank offset keeps DDP shuffles decorrelated).
    # Used by the Stage 5 seed replicates; absent → legacy unseeded behaviour.
    seed = opt.get('seed')
    if seed is not None:
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
        if rank == 0:
            print(f'Seeded RNGs (base {seed}, +rank offset)')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(opt['network'])

    pretrained  = opt_save.get('pretrained') or ''
    resume_path = opt_save.get('path', '')
    start_epoch = 0
    optim_state = sched_state = None

    if os.path.isfile(resume_path):
        model, optim_state, sched_state, start_epoch = load_weights(model, resume_path)
    elif pretrained and os.path.isfile(pretrained):
        model, *_ = load_weights(model, pretrained)
    elif rank == 0:
        print('No checkpoint found — training from scratch.')

    model.to(device)

    if ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # ── Losses ────────────────────────────────────────────────────────────────
    base_losses  = create_loss(opt, rank=device)
    phys_cfg     = opt.get('physics_losses', {})
    phys_enabled = phys_cfg.get('enabled', False)
    phys_losses, phys_extra_params = (
        create_physics_loss(phys_cfg, device) if phys_enabled else ({}, [])
    )

    # Restore learnable physics-loss params (blur PSF, etc.) on resume — these live in
    # the loss modules, not the model, so load_weights above does not cover them.
    if phys_enabled and phys_losses and start_epoch > 0 and os.path.isfile(resume_path):
        restore_phys_state(phys_losses, resume_path, device, rank)

    # ── Optimiser / scheduler ─────────────────────────────────────────────────
    raw_model  = model.module if ddp else model
    all_params = list(filter(lambda p: p.requires_grad, raw_model.parameters())) \
                 + phys_extra_params

    optimizer = torch.optim.AdamW(
        all_params,
        lr=opt_train['lr_initial'],
        weight_decay=opt_train['weight_decay'],
        betas=tuple(opt_train.get('betas', [0.9, 0.999])),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=opt_train['epochs'],
        eta_min=opt_train['eta_min'],
    )

    if optim_state:
        try:    optimizer.load_state_dict(optim_state)
        except Exception: pass
    if sched_state:
        try:    scheduler.load_state_dict(sched_state)
        except Exception: pass

    # ── EMA (rank 0 only — eval/save happen there) ────────────────────────────
    # The EMA copy is what we evaluate, select "best" on, and report. Restored on resume.
    ema_cfg     = opt_train.get('ema') or {}
    ema_enabled = ema_cfg.get('enabled', False)
    ema = None
    if ema_enabled and rank == 0:
        ema = ModelEMA(raw_model,
                       decay=ema_cfg.get('decay', 0.9999),
                       warmup=ema_cfg.get('warmup', 2000))
        if start_epoch > 0 and os.path.isfile(resume_path):
            try:
                _ck = torch.load(resume_path, map_location=device, weights_only=False)
                if 'ema' in _ck:
                    ema.load_state_dict(_ck['ema'])
                    ema.updates = _ck.get('ema_updates', 0)
                    print(f'  Restored EMA (updates={ema.updates})')
                else:
                    print('  No EMA in checkpoint — EMA re-seeded from resumed weights.')
                del _ck
            except Exception as e:
                print(f'  EMA restore failed ({e}) — re-seeded from resumed weights.')
        print(f'EMA enabled (decay={ema.decay}, warmup={ema.warmup}).')

    # eval / visual / final all run on the EMA weights when enabled
    eval_model = ema.ema if ema is not None else model

    # ── Data ──────────────────────────────────────────────────────────────────
    train_cfg = opt['datasets']['train']
    _SUPPORTED = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    low_paths  = _collect_images(train_cfg['low_path'])
    high_paths = _collect_images(train_cfg['high_path'])

    if len(low_paths) == 0:
        raise FileNotFoundError(f'No images found in {train_cfg["low_path"]}')
    if len(low_paths) != len(high_paths):
        raise ValueError(f'Mismatched image counts: {len(low_paths)} low vs {len(high_paths)} high')

    flips = transforms.RandomHorizontalFlip(p=0.5) if train_cfg.get('use_flips', True) else None
    train_dataset = MyDataset_Crop(
        images_low=low_paths,
        images_high=high_paths,
        cropsize=train_cfg.get('crop_size', 256),
        tensor_transform=transforms.ToTensor(),
        flips=flips,
        test=False,
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    ) if ddp else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_cfg.get('batch_size', 4),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=train_cfg.get('n_workers', 4),
        pin_memory=True,
        drop_last=True,
    )

    if rank == 0:
        print(f'Train images: {len(train_dataset)}  |  '
              f'Batches/epoch per rank: {len(train_loader)}  |  '
              f'World size: {world_size}')

    # Val loader — full dataset, rank 0 only, no DistributedSampler
    val_loader = None
    if rank == 0:
        try:
            val_loader, _ = create_test_data(rank=0, world_size=1, opt=opt['datasets'])
            print('Validation loader ready.')
        except Exception as e:
            print(f'Validation loader unavailable ({e}) — skipping eval.')

    # Generalization loader — optional out-of-distribution set (e.g. LOLv2).
    # Built from the `datasets_gen` config block; rank 0 only. Tracked separately
    # and never feeds best-checkpoint selection. Skip-safe if the block/path is absent.
    gen_loader = None
    gen_name   = None
    if rank == 0 and opt.get('datasets_gen'):
        gen_name = opt['datasets_gen'].get('name', 'gen')
        try:
            gen_loader, _ = create_test_data(rank=0, world_size=1, opt=opt['datasets_gen'])
            print(f'Generalization loader ready ({gen_name}).')
        except Exception as e:
            print(f'Generalization loader unavailable ({e}) — skipping gen eval.')

    # ── Checkpoint / logging setup (rank 0 only) ──────────────────────────────
    save_path = opt_save.get('path', './models/DarkIR_physics_latest.pt')
    best_path = opt_save.get('best', './models/bests/DarkIR_physics_best.pt')
    save_freq = opt_save.get('save_freq', 5)
    eval_freq = opt_save.get('eval_freq', 5)
    gen_eval_freq = opt_save.get('gen_eval_freq', 10)   # cadence for generalization (LOLv2) eval
    perceptual_epochs = set(opt_save.get('perceptual_eval_epochs', []))
    # Frozen mid-training snapshots (e.g. [150]) → '<save_path stem>_ep<N>.pt'.
    # Never overwritten afterwards — lets one run be evaluated at two operating points.
    snapshot_epochs = set(opt_save.get('snapshot_epochs', []))

    # Visual sample saving — off unless a 'visual_samples' block opts in.
    # Each run type sets its own epoch list (pilots/main/ablation differ in length).
    vis_cfg      = opt.get('visual_samples', {}) or {}
    vis_enabled  = vis_cfg.get('enabled', False)
    vis_epochs   = set(vis_cfg.get('epochs', []))
    vis_num      = vis_cfg.get('num_samples', 4)

    if rank == 0:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(best_path).parent.mkdir(parents=True, exist_ok=True)

    best_psnr  = 0.0
    best_epoch = start_epoch
    # Restore best-so-far on resume — otherwise best_psnr re-inits to 0 and the first
    # post-resume eval is wrongly stamped "best", overwriting the true best checkpoint.
    if start_epoch > 0 and os.path.isfile(resume_path):
        try:
            _bck = torch.load(resume_path, map_location='cpu', weights_only=False)
            best_psnr  = _bck.get('best_psnr', 0.0)
            best_epoch = _bck.get('best_epoch', start_epoch)
            if rank == 0 and best_psnr > 0:
                print(f'  Restored best-so-far: {best_psnr:.2f} dB @ epoch {best_epoch}')
            del _bck
        except Exception:
            pass
    grad_clip  = opt_train.get('grad_clip', 1.0)
    # Micro-batches per optimizer step (default 1 = plain training). Note: under DDP this
    # all-reduces every micro-batch (correct, just not comm-optimal) — fine for our runs.
    grad_accum = max(1, int(opt_train.get('grad_accum', 1)))
    if rank == 0 and grad_accum > 1:
        print(f'Gradient accumulation x{grad_accum} '
              f'(effective batch = loader batch x {grad_accum})')

    # Heavy eval models — rank 0 only
    lpips_fn          = None
    perceptual_metrics = {}
    if rank == 0:
        if _LPIPS_AVAILABLE:
            lpips_fn = _LPIPS(net='vgg', verbose=False).to(device)
        if perceptual_epochs and _PYIQA_AVAILABLE:
            print('Loading pyiqa metrics (NIQE, NRQM, MUSIQ)...')
            perceptual_metrics = {
                'niqe':  pyiqa.create_metric('niqe',  device=device, as_loss=False),
                'nrqm':  pyiqa.create_metric('nrqm',  device=device, as_loss=False),
                'musiq': pyiqa.create_metric('musiq', device=device, as_loss=False),
            }
        elif perceptual_epochs:
            print('Warning: perceptual_eval_epochs set but pyiqa not installed — skipping.')

    run_name = opt.get('wandb', {}).get('name', Path(config_path).stem)
    metrics_logger = MetricsLogger(run_name=run_name, config=opt) if rank == 0 else None

    def save_ckpt(epoch, is_best=False, snapshot_path=None):
        state = raw_model.state_dict()
        ckpt  = {
            'epoch'     : epoch,
            'model'     : state,
            'params'    : state,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
            'best_psnr' : best_psnr,   # restored on resume so best-checkpoint selection survives
            'best_epoch': best_epoch,
        }
        if phys_enabled and phys_losses:
            # Full module state for correct resume (log_sigma etc.); blur_sigma kept for logs.
            ckpt['phys_state'] = {n: m.state_dict() for n, m in phys_losses.items()}
            if 'blur_loss' in phys_losses:
                ckpt['blur_sigma'] = phys_losses['blur_loss'].log_sigma.item()
        if ema is not None:
            # EMA weights = the reported/deployable model; 'model' stays online for resume.
            ckpt['ema']         = ema.state_dict()
            ckpt['ema_updates'] = ema.updates
        # Atomic writes: tmp + os.replace, so a session killed mid-save can never
        # leave a corrupt checkpoint behind — the previous complete file survives.
        def _atomic_save(obj, path):
            tmp = str(path) + '.tmp'
            torch.save(obj, tmp)
            os.replace(tmp, path)
        _atomic_save(ckpt, save_path)
        if is_best:
            _atomic_save(ckpt, best_path)
        if snapshot_path:
            _atomic_save(ckpt, snapshot_path)

    # ── Optional wandb (rank 0) ───────────────────────────────────────────────
    use_wandb = False
    if rank == 0 and opt.get('wandb', {}).get('init', False):
        try:
            import wandb as _wandb
            _wandb.login()
            _wandb.init(
                project=opt['wandb']['project'],
                entity=opt['wandb'].get('entity'),
                name=opt['wandb'].get('name', 'physics_finetune'),
                config=opt,
                resume=opt['wandb'].get('resume', False),
                id=opt['wandb'].get('id'),
                dir=opt['wandb'].get('dir', './logs'),
            )
            use_wandb = True
        except Exception as e:
            print(f'wandb init failed ({e}) — continuing without logging.')

    # ── Training loop ─────────────────────────────────────────────────────────
    total_epochs = opt_train['epochs']
    prev_phase   = 0
    ep           = {}   # populated each epoch; needed for finalize

    # Optional wall-clock budget (hours). Stops cleanly after a full epoch + save,
    # so a 200-ep run can span several capped sessions (Kaggle 12 h) via resume.
    time_budget_s = float(opt_train.get('time_budget_h', 0) or 0) * 3600
    t_train_start = time.time()

    if rank == 0:
        print(f'\nStarting from epoch {start_epoch + 1} / {total_epochs}')
        if time_budget_s:
            print(f'Wall-clock budget: {time_budget_s/3600:.2f} h (clean stop + resume)')

    for epoch in range(start_epoch + 1, total_epochs + 1):

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)   # ensures different shuffles per epoch

        model.train()
        phase = get_phase(epoch, phys_cfg) if phys_enabled else 1

        if rank == 0 and phase != prev_phase:
            print(f'\n── Phase {phase} (epoch {epoch}) ── {_PHASE_LABELS[phase]}')
            prev_phase = phase

        running = dict(total=0., base=0., retinex=0., blur=0., phase_freq=0.)

        pbar = tqdm(train_loader, desc=f'Ep {epoch}/{total_epochs}',
                    leave=False, dynamic_ncols=True, disable=(rank != 0))

        n_batches = len(train_loader)
        optimizer.zero_grad()
        for bi, (high_batch, low_batch) in enumerate(pbar):
            high_batch = high_batch.to(device)
            low_batch  = low_batch.to(device)

            enhanced = model(low_batch, side_loss=False)
            enhanced = torch.clamp(enhanced, 0., 1.)

            loss   = calculate_loss(base_losses, enhanced, high_batch)
            l_base = loss.item()

            l_retinex = l_blur = l_phase = 0.

            if phys_enabled and phase >= 2 and 'retinex_loss' in phys_losses:
                lr = phys_losses['retinex_loss'](enhanced, high_batch)
                loss      = loss + lr
                l_retinex = lr.item()

            if phys_enabled and phase >= 3 and 'blur_loss' in phys_losses:
                lb     = phys_losses['blur_loss'](enhanced, low_batch)
                loss   = loss + lb
                l_blur = lb.item()

            if phys_enabled and phase >= 4 and 'phase_loss' in phys_losses:
                lp      = phys_losses['phase_loss'](enhanced, high_batch)
                loss    = loss + lp
                l_phase = lp.item()

            # Gradient accumulation: summed micro-batch grads / grad_accum = the exact
            # mean-gradient of the full effective batch, so batch 16 x accum 2 reproduces
            # batch-32 training (same steps/epoch, same EMA update count) in less memory.
            (loss / grad_accum).backward()

            if (bi + 1) % grad_accum == 0 or (bi + 1) == n_batches:
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(all_params, max_norm=grad_clip)
                optimizer.step()
                if ema is not None:
                    ema.update(raw_model)
                optimizer.zero_grad()

            n = high_batch.size(0)
            running['total']      += loss.item() * n
            running['base']       += l_base       * n
            running['retinex']    += l_retinex     * n
            running['blur']       += l_blur        * n
            running['phase_freq'] += l_phase       * n

            if rank == 0:
                pbar.set_postfix(loss=f'{loss.item():.4f}',
                                 lr=f'{scheduler.get_last_lr()[0]:.2e}')

        scheduler.step()

        # ── Reduce losses across ranks → true dataset-mean on all ranks ───────
        N = len(train_dataset)  # total images (same on all ranks)
        if ddp:
            for key in running:
                running[key] = reduce_scalar(running[key], device)
        ep = {k: v / N for k, v in running.items()}
        ep['epoch'] = epoch
        ep['lr']    = scheduler.get_last_lr()[0]

        blur_sigma_val = None
        if phys_enabled and 'blur_loss' in phys_losses:
            blur_sigma_val = phys_losses['blur_loss'].log_sigma.exp().item()
            ep['blur_sigma'] = blur_sigma_val

        # ── Rank-0: logging, eval, checkpointing ──────────────────────────────
        if rank == 0:
            print(f'Epoch {epoch:3d} | total={ep["total"]:.4f}  base={ep["base"]:.4f}  '
                  f'retinex={ep["retinex"]:.4f}  blur={ep["blur"]:.4f}  '
                  f'phase={ep["phase_freq"]:.4f}  lr={ep["lr"]:.2e}'
                  + (f'  σ={ep["blur_sigma"]:.3f}' if 'blur_sigma' in ep else ''))

            if epoch == 1:
                assert ep['total'] < 10.0, \
                    f"loss_total={ep['total']:.4f} at epoch 1 — check data paths or loss weights"
                assert ep['base'] > 0., \
                    "base loss is zero at epoch 1 — model output may be detached"
                print(f'  ✅ Epoch 1 sanity check passed (loss_total={ep["total"]:.4f})')

            if phys_enabled and phase >= 2 and ep['total'] > 0:
                for name in ('base', 'retinex', 'blur', 'phase_freq'):
                    share = ep[name] / ep['total']
                    if share > 0.60 and ep[name] > 0:
                        print(f'  ⚠️  {name} is {share*100:.0f}% of total loss '
                              f'— consider lowering its weight')

            is_best    = False
            mean_psnr  = mean_ssim = mean_lpips = None

            if val_loader is not None and epoch % eval_freq == 0 and lpips_fn is not None:
                try:
                    loaders = val_loader.items() if isinstance(val_loader, dict) \
                              else [('val', val_loader)]
                    all_psnr, all_ssim, all_lpips = [], [], []
                    for key, ld in loaders:
                        loader = ld['loader'] if isinstance(ld, dict) else ld
                        m = eval_metrics(eval_model, loader, device, lpips_fn)
                        all_psnr.append(m['psnr'])
                        all_ssim.append(m['ssim'])
                        all_lpips.append(m['lpips'])
                        print(f'  val [{key}] PSNR={m["psnr"]:.2f} dB  '
                              f'SSIM={m["ssim"]:.4f}  LPIPS={m["lpips"]:.4f}')
                    mean_psnr  = float(np.mean(all_psnr))
                    mean_ssim  = float(np.mean(all_ssim))
                    mean_lpips = float(np.mean(all_lpips))
                    if mean_psnr > best_psnr:
                        best_psnr  = mean_psnr
                        best_epoch = epoch
                        is_best    = True
                        print(f'  New best PSNR: {best_psnr:.2f} dB')
                    ep['val_psnr']  = mean_psnr
                    ep['val_ssim']  = mean_ssim
                    ep['val_lpips'] = mean_lpips
                except Exception as e:
                    print(f'  Eval failed: {e}')

            # Generalization eval (e.g. LOLv2) — separate cadence, separate log,
            # never feeds best-checkpoint selection. Tracks OOD generalization only.
            # gen_loader may be a dict of sub-loaders (All_LOL → lolv2 / lolv2_synth / …)
            # or a single loader; each sub-set is logged under its own dataset key.
            if gen_loader is not None and epoch % gen_eval_freq == 0 and lpips_fn is not None:
                try:
                    gen_loaders = gen_loader.items() if isinstance(gen_loader, dict) \
                                  else [(gen_name, gen_loader)]
                    for key, ld in gen_loaders:
                        loader = ld['loader'] if isinstance(ld, dict) else ld
                        g = eval_metrics(eval_model, loader, device, lpips_fn)
                        print(f'  gen [{key}] PSNR={g["psnr"]:.2f} dB  '
                              f'SSIM={g["ssim"]:.4f}  LPIPS={g["lpips"]:.4f}')
                        metrics_logger.log_generalization(
                            epoch=epoch, phase=phase, dataset=key,
                            val_psnr=g['psnr'], val_ssim=g['ssim'], val_lpips=g['lpips'],
                        )
                except Exception as e:
                    print(f'  Gen eval failed: {e}')

            if epoch in perceptual_epochs and perceptual_metrics and val_loader is not None:
                try:
                    loader = list(val_loader.values())[0]['loader'] \
                             if isinstance(val_loader, dict) else val_loader
                    perc = eval_perceptual(eval_model, loader, device, perceptual_metrics)
                    print(f'  perceptual [ep {epoch}] '
                          f'NIQE={perc["niqe"]:.3f}↓  '
                          f'NRQM={perc["nrqm"]:.3f}↑  '
                          f'MUSIQ={perc["musiq"]:.3f}↑')
                    metrics_logger.log_perceptual(epoch=epoch, phase=phase, scores=perc)
                except Exception as e:
                    print(f'  Perceptual eval failed: {e}')

            if vis_enabled and epoch in vis_epochs and val_loader is not None:
                try:
                    loader = list(val_loader.values())[0]['loader'] \
                             if isinstance(val_loader, dict) else val_loader
                    sample_dir = Path('.metrics') / run_name / 'samples'
                    p = save_visual_samples(eval_model, loader, device, sample_dir,
                                            epoch, num_samples=vis_num)
                    if p:
                        print(f'  Saved {vis_num} visual samples → {p}')
                except Exception as e:
                    print(f'  Visual sample save failed: {e}')

            metrics_logger.log_epoch(
                epoch=epoch, phase=phase,
                losses={k: ep[k] for k in ('total', 'base', 'retinex', 'blur', 'phase_freq')},
                val_psnr=mean_psnr, val_ssim=mean_ssim, val_lpips=mean_lpips,
                lr=ep['lr'], blur_sigma=blur_sigma_val,
            )

            snap_path = None
            if epoch in snapshot_epochs:
                _sp = Path(save_path)
                snap_path = str(_sp.with_name(f'{_sp.stem}_ep{epoch}{_sp.suffix}'))
            if epoch % save_freq == 0 or is_best or snap_path:
                save_ckpt(epoch, is_best=is_best, snapshot_path=snap_path)
                print(f'  Checkpoint saved (epoch {epoch}'
                      f'{"  [best]" if is_best else ""}'
                      f'{"  [snapshot]" if snap_path else ""})')
                metrics_logger.log_checkpoint(
                    epoch=epoch, phase=phase,
                    losses={k: ep[k] for k in ('total', 'base', 'retinex', 'blur', 'phase_freq')},
                    val_psnr=mean_psnr, val_ssim=mean_ssim, val_lpips=mean_lpips,
                    is_best=is_best, blur_sigma=blur_sigma_val,
                )

            if use_wandb:
                import wandb as _wandb
                _wandb.log(ep)

        # All ranks sync before next epoch
        if ddp:
            dist.barrier()

        # Budget check — rank 0 decides, all ranks follow (avoids DDP desync).
        if time_budget_s:
            _stop = torch.tensor(
                [1.0 if (time.time() - t_train_start) > time_budget_s else 0.0],
                device=device)
            if ddp:
                dist.broadcast(_stop, src=0)
            if _stop.item():
                if rank == 0:
                    save_ckpt(epoch)
                    print(f'\nTime budget reached at epoch {epoch} — checkpoint saved, '
                          f'stopping cleanly (re-run to resume).')
                break

    # ── End of training ───────────────────────────────────────────────────────
    if rank == 0:
        metrics_logger.finalize(
            best_epoch=best_epoch, best_psnr=best_psnr,
            best_ssim=ep.get('val_ssim'), best_lpips=ep.get('val_lpips'),
        )
        print('\nTraining complete.')
        if use_wandb:
            import wandb as _wandb
            _wandb.finish()

    cleanup_ddp()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DarkIR physics training')
    parser.add_argument('-p', '--config', type=str,
                        default='./options/train/physics_finetune.yml')
    args = parser.parse_args()

    # torchrun sets LOCAL_RANK; fall back to single-device when absent
    local_rank  = int(os.environ.get('LOCAL_RANK', 0))
    world_size  = int(os.environ.get('WORLD_SIZE', 1))

    train_worker(rank=local_rank, world_size=world_size, config_path=args.config)


