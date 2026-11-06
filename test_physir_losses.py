"""
Stage 0 unit tests for PhysIR physics-guided loss functions.

Tests the ACTUAL API (not the guide's assumed API — see .plans/STAGE0_PLAN.md for diffs):
  - RetinexLoss.forward(x_pred)                         [self-supervised, no target]
  - PhaseEnhancedFrequencyLoss.forward(pred, target)    [supervised, vs GT]
  - BlurAwareGradientLoss.forward(x_pred, y_input)      [self-supervised, vs blurry input]

Run: python test_physir_losses.py
"""

import torch
import torch.nn.functional as F


def test_retinex_loss():
    from losses.loss import RetinexLoss

    loss_fn = RetinexLoss(weight=0.1, smooth_weight=0.5, recon_weight=0.5)

    pred   = torch.rand(2, 3, 64, 64, requires_grad=True)
    target = torch.rand(2, 3, 64, 64)

    # Basic compute
    loss = loss_fn(pred, target)
    assert not torch.isnan(loss),  "RetinexLoss returned NaN"
    assert not torch.isinf(loss),  "RetinexLoss returned Inf"
    assert loss.item() > 0,        "RetinexLoss is zero (suspicious)"

    # Gradients flow
    loss.backward()
    assert pred.grad is not None,                    "No gradients to pred"
    assert not torch.isnan(pred.grad).any(),         "NaN in gradients"

    # Recon term is non-trivial: pred == target should give lower recon than random pred
    pred_exact = target.clone().detach().requires_grad_(True)
    loss_exact = loss_fn(pred_exact, target)
    assert loss_exact.item() < loss.item(), \
        "pred==target should have lower RetinexLoss than random pred"

    # Smoothness: uniform pred should have lower TV term than noisy pred
    # (fix target to isolate the TV term)
    uniform_pred = torch.ones(2, 3, 64, 64) * 0.5
    noisy_pred   = torch.rand(2, 3, 64, 64)
    same_target  = torch.ones(2, 3, 64, 64) * 0.5  # same as uniform → recon ~0 for both
    loss_uniform = loss_fn(uniform_pred, same_target)
    loss_noisy   = loss_fn(noisy_pred,   same_target)
    assert loss_uniform.item() < loss_noisy.item(), \
        "Uniform pred should have lower TV loss than random pred"

    # Inspect log-domain reflectances
    eps = 1e-4
    with torch.no_grad():
        pool  = torch.nn.MaxPool2d(15, stride=1, padding=7)
        L     = pool(pred.detach().max(dim=1, keepdim=True).values).clamp(min=eps)
        logR  = torch.log(pred.detach() + eps) - torch.log(L.expand_as(pred.detach()) + eps)
        print(f"  log(R) range: [{logR.min():.3f}, {logR.max():.3f}]  (centred near 0)")

    print(f"  loss value (random pred):  {loss.item():.4f}")
    print(f"  loss value (pred==target): {loss_exact.item():.6f}  (should be near 0)")
    print(f"  grad norm:  {pred.grad.norm().item():.4f}")
    print("✅ RetinexLoss passed\n")


def test_phase_frequency_loss():
    from losses.loss import PhaseEnhancedFrequencyLoss

    loss_fn = PhaseEnhancedFrequencyLoss(weight=0.05, amp_weight=0.5, phase_weight=0.5)

    pred   = torch.rand(2, 3, 64, 64, requires_grad=True)
    target = torch.rand(2, 3, 64, 64)

    # Basic compute
    loss = loss_fn(pred, target)
    assert not torch.isnan(loss), "PhaseEnhancedFrequencyLoss returned NaN"
    assert not torch.isinf(loss), "PhaseEnhancedFrequencyLoss returned Inf"
    assert loss.item() > 0,       "PhaseEnhancedFrequencyLoss is zero (suspicious)"

    # Gradients flow
    loss.backward()
    assert pred.grad is not None,                    "No gradients to pred"
    assert not torch.isnan(pred.grad).any(),         "NaN in gradients"

    # Similar images → lower loss
    pred_similar = target + 0.01 * torch.randn_like(target)
    pred_similar.requires_grad_(True)
    loss_similar = loss_fn(pred_similar, target)
    assert loss_similar.item() < loss.item(), \
        "Similar images should have lower PhaseEnhancedFrequencyLoss"

    # pred == target → near-zero
    pred_exact = target.clone().detach().requires_grad_(True)
    loss_exact = loss_fn(pred_exact, target)
    assert loss_exact.item() < 1e-3, \
        f"Identical pred/target should give near-zero loss, got {loss_exact.item():.6f}"

    print(f"  loss (random pair):  {loss.item():.4f}")
    print(f"  loss (similar pair): {loss_similar.item():.4f}  (should be lower)")
    print(f"  loss (exact match):  {loss_exact.item():.6f}  (should be ~0)")
    print("✅ PhaseEnhancedFrequencyLoss passed\n")


def test_blur_aware_gradient_loss():
    from losses.loss import BlurAwareGradientLoss

    loss_fn = BlurAwareGradientLoss(weight=0.05, init_sigma=1.5, kernel_size=11)

    x_pred  = torch.rand(2, 3, 64, 64, requires_grad=True)
    y_input = torch.rand(2, 3, 64, 64)  # blurry/degraded input image

    # Basic compute
    loss = loss_fn(x_pred, y_input)
    assert not torch.isnan(loss), "BlurAwareGradientLoss returned NaN"
    assert not torch.isinf(loss), "BlurAwareGradientLoss returned Inf"
    assert loss.item() > 0,       "BlurAwareGradientLoss is zero (suspicious)"

    # Gradients flow
    loss.backward()
    assert x_pred.grad is not None,                    "No gradients to x_pred"
    assert not torch.isnan(x_pred.grad).any(),         "NaN in gradients"

    # Learnable sigma attribute exists and is in valid range
    sigma = loss_fn.log_sigma.exp()
    assert sigma.item() > 0,    "Sigma must be positive"
    assert sigma.item() < 10.0, "Sigma seems too large"

    # Optimizer can see it as a parameter
    assert loss_fn.log_sigma in list(loss_fn.parameters()), \
        "log_sigma must be a registered nn.Parameter"

    # Simulate an optimizer step — sigma should update
    optim = torch.optim.SGD(loss_fn.parameters(), lr=0.1)
    sigma_before = loss_fn.log_sigma.item()
    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    y = torch.rand(1, 3, 32, 32)
    loss_fn(x, y).backward()
    optim.step()
    sigma_after = loss_fn.log_sigma.item()
    assert abs(sigma_after - sigma_before) > 1e-9, \
        "log_sigma did not update after optimizer step — not learning"

    print(f"  loss value:    {loss.item():.4f}")
    print(f"  initial σ:     {sigma.item():.4f}  (will be learned during training)")
    print(f"  σ after step:  {loss_fn.log_sigma.exp().item():.4f}  (changed: ✅)")
    print("✅ BlurAwareGradientLoss passed\n")


def test_create_physics_loss_factory():
    """Verify the factory function wires up all three losses correctly."""
    from losses import create_physics_loss

    opt = {
        'retinex':        True,
        'retinex_weight': 0.1,
        'blur':           True,
        'blur_weight':    0.05,
        'phase':          True,
        'phase_weight':   0.05,
    }
    losses, extra_params = create_physics_loss(opt, 'cpu')

    assert 'retinex_loss' in losses, "retinex_loss not in factory output"
    assert 'blur_loss'    in losses, "blur_loss not in factory output"
    assert 'phase_loss'   in losses, "phase_loss not in factory output"
    assert len(extra_params) > 0,    "extra_params should contain blur sigma"

    # All losses run without error on dummy tensors
    pred  = torch.rand(1, 3, 32, 32)
    gt    = torch.rand(1, 3, 32, 32)
    blurry = torch.rand(1, 3, 32, 32)

    l_r = losses['retinex_loss'](pred, gt)
    l_b = losses['blur_loss'](pred, blurry)
    l_p = losses['phase_loss'](pred, gt)

    for name, val in [('retinex', l_r), ('blur', l_b), ('phase', l_p)]:
        assert not torch.isnan(val), f"{name} loss is NaN via factory"
        assert not torch.isinf(val), f"{name} loss is Inf via factory"

    print(f"  retinex: {l_r.item():.4f}  blur: {l_b.item():.4f}  phase: {l_p.item():.4f}")
    print("✅ create_physics_loss factory passed\n")


if __name__ == '__main__':
    print("\n=== PhysIR Loss Unit Tests ===\n")

    print("[RetinexLoss]")
    test_retinex_loss()

    print("[PhaseEnhancedFrequencyLoss]")
    test_phase_frequency_loss()

    print("[BlurAwareGradientLoss]")
    test_blur_aware_gradient_loss()

    print("[Factory: create_physics_loss]")
    test_create_physics_loss_factory()

    print("✅ All unit tests passed. Ready for Stage 1 pilots.")
    print("   Next: implement train.py and YML configs (see .plans/STAGE0_PLAN.md)\n")
