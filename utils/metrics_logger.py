"""
MetricsLogger — writes training metrics to .metrics/<run_name>/ at each epoch
and checkpoint. Produces three artefacts per run:

  training_log.csv      — one row per epoch; machine-readable for plotting/diffing
  training_log.md       — same data as a markdown table; readable in GitHub/editors
  checkpoint_summary.md — one entry per checkpoint + run header with config snapshot

Ablation comparison:
  grep "best_psnr" .metrics/*/checkpoint_summary.md | sort -t: -k2 -rn | head -5
  paste -d',' .metrics/pilot_A_baseline/training_log.csv \\
              .metrics/pilot_B_retinex/training_log.csv
"""

import csv
import json
from datetime import datetime
from pathlib import Path


_CSV_FIELDS = [
    'epoch', 'phase',
    'loss_total', 'loss_base', 'loss_retinex', 'loss_blur', 'loss_phase_freq',
    'val_psnr', 'val_ssim', 'val_lpips', 'lr', 'blur_sigma',
]

_MD_HEADER = (
    '| epoch | phase | total | base | retinex | blur | phase_freq '
    '| psnr | ssim | lpips | lr | blur_σ |\n'
    '|------:|------:|------:|-----:|--------:|-----:|----------:'
    '|-----:|-----:|------:|---:|-------:|\n'
)


class MetricsLogger:

    def __init__(self, run_name: str, config: dict, metrics_dir='.metrics'):
        self.run_name  = run_name
        self.run_dir   = Path(metrics_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path     = self.run_dir / 'training_log.csv'
        self.md_path      = self.run_dir / 'training_log.md'
        self.summary_path = self.run_dir / 'checkpoint_summary.md'

        self._write_config(config)
        self._init_files()

    # ── setup ─────────────────────────────────────────────────────────────────

    def _write_config(self, config):
        with open(self.run_dir / 'run_config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)

    def _init_files(self):
        self.perceptual_path = self.run_dir / 'perceptual_log.csv'
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()
        if not self.md_path.exists():
            with open(self.md_path, 'w') as f:
                f.write(f'# Training log — {self.run_name}\n\n')
                f.write(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n\n')
                f.write(_MD_HEADER)
        if not self.summary_path.exists():
            with open(self.summary_path, 'w') as f:
                f.write(f'# Checkpoint summary — {self.run_name}\n\n')
                f.write(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n\n')

    # ── perceptual call ────────────────────────────────────────────────────────

    def log_perceptual(self, epoch: int, phase: int, scores: dict):
        """
        Call at perceptual_eval_epochs only. Writes to perceptual_log.csv.
        scores: dict with keys niqe (↓), nrqm (↑), musiq (↑).
        Kept separate from training_log.csv to avoid wide sparse columns.
        """
        _PERC_FIELDS = ['epoch', 'phase', 'niqe', 'nrqm', 'musiq']
        write_header = not self.perceptual_path.exists()
        row = {
            'epoch': epoch, 'phase': phase,
            'niqe':  f"{scores.get('niqe',  float('nan')):.4f}",
            'nrqm':  f"{scores.get('nrqm',  float('nan')):.4f}",
            'musiq': f"{scores.get('musiq', float('nan')):.4f}",
        }
        with open(self.perceptual_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=_PERC_FIELDS)
            if write_header:
                w.writeheader()
            w.writerow(row)

        # Also append a block to checkpoint_summary for at-a-glance reading
        with open(self.summary_path, 'a') as f:
            f.write(
                f'\n### Perceptual eval — epoch {epoch}\n'
                f'| Metric | Score | Direction |\n'
                f'|--------|------:|-----------|\n'
                f'| NIQE  | {scores.get("niqe",  float("nan")):.4f} | ↓ lower = better |\n'
                f'| NRQM  | {scores.get("nrqm",  float("nan")):.4f} | ↑ higher = better |\n'
                f'| MUSIQ | {scores.get("musiq", float("nan")):.4f} | ↑ higher = better |\n'
            )

    # ── generalization call ─────────────────────────────────────────────────────

    def log_generalization(self, epoch: int, phase: int, dataset: str,
                           val_psnr: float = None, val_ssim: float = None,
                           val_lpips: float = None):
        """
        Call at gen_eval_freq epochs only. Writes to generalization_log.csv.
        Tracks out-of-distribution performance (e.g. LOLv2) WITHOUT feeding into
        best-checkpoint selection. Kept separate from training_log.csv so the
        in-distribution (LOLBlur) curves stay clean.
        """
        self.generalization_path = self.run_dir / 'generalization_log.csv'
        _GEN_FIELDS = ['epoch', 'phase', 'dataset', 'val_psnr', 'val_ssim', 'val_lpips']
        write_header = not self.generalization_path.exists()
        row = {
            'epoch': epoch, 'phase': phase, 'dataset': dataset,
            'val_psnr':  f"{val_psnr:.4f}"  if val_psnr  is not None else '',
            'val_ssim':  f"{val_ssim:.4f}"  if val_ssim  is not None else '',
            'val_lpips': f"{val_lpips:.4f}" if val_lpips is not None else '',
        }
        with open(self.generalization_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=_GEN_FIELDS)
            if write_header:
                w.writeheader()
            w.writerow(row)

    # ── per-epoch call ─────────────────────────────────────────────────────────

    def log_epoch(self, epoch: int, phase: int, losses: dict,
                  val_psnr: float = None, val_ssim: float = None,
                  val_lpips: float = None, lr: float = 0.,
                  blur_sigma: float = None):
        row = {
            'epoch':           epoch,
            'phase':           phase,
            'loss_total':      f"{losses.get('total',     0.):.6f}",
            'loss_base':       f"{losses.get('base',      0.):.6f}",
            'loss_retinex':    f"{losses.get('retinex',   0.):.6f}",
            'loss_blur':       f"{losses.get('blur',      0.):.6f}",
            'loss_phase_freq': f"{losses.get('phase_freq',0.):.6f}",
            'val_psnr':        f"{val_psnr:.4f}"  if val_psnr  is not None else '',
            'val_ssim':        f"{val_ssim:.4f}"  if val_ssim  is not None else '',
            'val_lpips':       f"{val_lpips:.4f}" if val_lpips is not None else '',
            'lr':              f"{lr:.2e}",
            'blur_sigma':      f"{blur_sigma:.4f}" if blur_sigma is not None else '',
        }
        with open(self.csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerow(row)

        def _fmt(v, fmt): return fmt % v if v is not None else '—'
        with open(self.md_path, 'a') as f:
            f.write(
                f"| {epoch:5d} | {phase} "
                f"| {losses.get('total',0.):.4f} "
                f"| {losses.get('base',0.):.4f} "
                f"| {losses.get('retinex',0.):.4f} "
                f"| {losses.get('blur',0.):.4f} "
                f"| {losses.get('phase_freq',0.):.4f} "
                f"| {_fmt(val_psnr,  '%.2f'):>6} "
                f"| {_fmt(val_ssim,  '%.4f'):>6} "
                f"| {_fmt(val_lpips, '%.4f'):>6} "
                f"| {lr:.1e} "
                f"| {_fmt(blur_sigma,'%.3f')} |\n"
            )

    # ── checkpoint call ────────────────────────────────────────────────────────

    def log_checkpoint(self, epoch: int, phase: int, losses: dict,
                       val_psnr: float = None, val_ssim: float = None,
                       val_lpips: float = None, is_best: bool = False,
                       blur_sigma: float = None):
        def _fmt(v, fmt): return fmt % v if v is not None else 'not evaluated'
        best_tag  = '  ⭐ **BEST**' if is_best else ''
        ts        = datetime.now().strftime('%Y-%m-%d %H:%M')
        balance   = _loss_balance_str(losses)

        block = (
            f'\n## Epoch {epoch}{best_tag}\n'
            f'_Saved: {ts}_\n\n'
            f'| Metric | Value |\n'
            f'|--------|-------|\n'
            f'| Phase | {phase} ({_phase_label(phase)}) |\n'
            f'| val PSNR  | {_fmt(val_psnr,  "%.4f dB")} |\n'
            f'| val SSIM  | {_fmt(val_ssim,  "%.4f")} |\n'
            f'| val LPIPS | {_fmt(val_lpips, "%.4f")} |\n'
            f'| blur σ | {_fmt(blur_sigma, "%.4f")} |\n'
            f'| loss_total | {losses.get("total", 0.):.6f} |\n'
            f'| loss_base | {losses.get("base", 0.):.6f} |\n'
            f'| loss_retinex | {losses.get("retinex", 0.):.6f} |\n'
            f'| loss_blur | {losses.get("blur", 0.):.6f} |\n'
            f'| loss_phase_freq | {losses.get("phase_freq", 0.):.6f} |\n\n'
            f'**Loss balance:**\n```\n{balance}\n```\n'
        )
        with open(self.summary_path, 'a') as f:
            f.write(block)

    # ── final call ─────────────────────────────────────────────────────────────

    def finalize(self, best_epoch: int, best_psnr: float,
                 best_ssim: float = None, best_lpips: float = None):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M')
        def _fmt(v, fmt): return fmt % v if v is not None else '—'
        block = (
            f'\n---\n'
            f'## Run complete\n'
            f'_Finished: {ts}_\n\n'
            f'| | |\n|---|---|\n'
            f'| Best epoch | {best_epoch} |\n'
            f'| Best val PSNR  | {best_psnr:.4f} dB |\n'
            f'| Best val SSIM  | {_fmt(best_ssim,  "%.4f")} |\n'
            f'| Best val LPIPS | {_fmt(best_lpips, "%.4f")} |\n'
        )
        with open(self.summary_path, 'a') as f:
            f.write(block)
        with open(self.md_path, 'a') as f:
            f.write(f'\n_Run complete {ts} — best: epoch {best_epoch}, '
                    f'PSNR {best_psnr:.4f} dB_\n')

    @staticmethod
    def loss_balance_str(losses: dict) -> str:
        return _loss_balance_str(losses)


# ── module-level helpers ───────────────────────────────────────────────────────

def _phase_label(phase: int) -> str:
    return {1: 'L1+Edge', 2: '+Retinex', 3: '+BlurGrad',
            4: '+PhaseFreq [FULL]'}.get(phase, '?')


def _loss_balance_str(losses: dict, width: int = 50) -> str:
    total = sum(v for v in losses.values() if isinstance(v, float))
    if total == 0:
        return 'total=0 (no loss data)'
    lines = []
    for name, val in losses.items():
        if not isinstance(val, float):
            continue
        pct = 100.0 * val / total
        bar = '█' * max(0, int(pct * width / 100))
        lines.append(f'  {name:<14s}: {val:8.4f} ({pct:5.1f}%) {bar}')
    lines.append(f'  {"TOTAL":<14s}: {total:8.4f}')
    return '\n'.join(lines)
