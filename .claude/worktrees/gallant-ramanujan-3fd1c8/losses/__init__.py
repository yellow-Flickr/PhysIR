from .loss import (MSELoss, L1Loss, CharbonnierLoss, SSIM, VGGLoss,
                   EdgeLoss, FrequencyLoss, EnhanceLoss,
                   RetinexLoss, BlurAwareGradientLoss, PhaseEnhancedFrequencyLoss)


def create_loss(opt, rank):
    '''
    Returns the needed losses for evaluating our model.
    opt: dictionary from the yaml config root.
    '''
    losses = dict()

    if opt['pixel_criterion'] == 'l1':
        pixel_loss = L1Loss()
    elif opt['pixel_criterion'] == 'l2':
        pixel_loss = MSELoss()
    elif opt['pixel_criterion'] == 'Charbonnier':
        pixel_loss = CharbonnierLoss()
    else:
        raise NotImplementedError('Pixel Criterion not implemented')

    losses['pixel_loss'] = pixel_loss.to(rank)
    if rank == 0: print(f"Using pixel loss {opt['pixel_criterion']}")

    if opt['perceptual']:
        perceptual_loss = VGGLoss(loss_weight=opt['perceptual_weight'],
                                  criterion=opt['perceptual_criterion'],
                                  reduction=opt['perceptual_reduction']).to(rank)
        losses['perceptual_loss'] = perceptual_loss
        if rank == 0: print(f"Using perceptual loss {opt['perceptual_criterion']} "
                            f"weight {opt['perceptual_weight']}")

    if opt['edge']:
        edge_loss = EdgeLoss(loss_weight=opt['edge_weight'],
                             criterion=opt['edge_criterion'],
                             reduction=opt['edge_reduction'],
                             rank=rank).to(rank)
        losses['edge_loss'] = edge_loss
        if rank == 0: print(f"Using edge loss {opt['edge_criterion']} "
                            f"weight {opt['edge_weight']}")

    if opt['frequency']:
        frequency_loss = FrequencyLoss(loss_weight=opt['frequency_weight'],
                                       reduction=opt['edge_reduction'],
                                       criterion=opt['frequency_criterion']).to(rank)
        losses['frequency_loss'] = frequency_loss
        if rank == 0: print(f"Using frequency loss {opt['frequency_criterion']} "
                            f"weight {opt['frequency_weight']}")

    if opt['enhance']:
        enhance_loss = EnhanceLoss(loss_weight=opt['enhance_weight'],
                                   criterion=opt['enhance_criterion'],
                                   reduction=opt['enhance_reduction']).to(rank)
        losses['enhance_loss'] = enhance_loss
        if rank == 0: print(f"Using enhance loss {opt['enhance_criterion']} "
                            f"weight {opt['enhance_weight']}")

    return losses


def create_physics_loss(opt, device):
    '''
    Instantiates physics-guided losses from the physics_losses config block.

    Returns:
        losses      : dict of loss modules (already moved to device)
        extra_params: list of extra trainable params (learnable blur sigma)
                      — pass to the optimizer alongside model.parameters()
    '''
    losses       = {}
    extra_params = []

    if opt.get('retinex', True):
        rl = RetinexLoss(weight=opt.get('retinex_weight', 0.10)).to(device)
        losses['retinex_loss'] = rl

    if opt.get('blur', True):
        bl = BlurAwareGradientLoss(weight=opt.get('blur_weight', 0.05)).to(device)
        losses['blur_loss'] = bl
        extra_params += list(bl.parameters())  # includes learnable sigma

    if opt.get('phase', True):
        pl = PhaseEnhancedFrequencyLoss(weight=opt.get('phase_weight', 0.05)).to(device)
        losses['phase_loss'] = pl

    return losses, extra_params


def calculate_loss(all_losses, enhanced_batch, high_batch,
                   outside_batch=None, scale_factor=8):
    '''
    Sums all active supervised losses.
    outside_batch: optional side-output from the encoder mid-point.
    '''
    l_pixel = all_losses['pixel_loss'](enhanced_batch, high_batch)
    if 'perceptual_loss' in all_losses:
        l_pixel += all_losses['perceptual_loss'](enhanced_batch, high_batch)
    if 'edge_loss' in all_losses:
        l_pixel += all_losses['edge_loss'](enhanced_batch, high_batch)
    if 'frequency_loss' in all_losses:
        l_pixel += all_losses['frequency_loss'](enhanced_batch, high_batch)
    if 'enhance_loss' in all_losses and outside_batch is not None:
        l_pixel += all_losses['enhance_loss'](outside_batch, high_batch,
                                              scale_factor=scale_factor)
    return l_pixel


__all__ = ['create_loss', 'create_physics_loss', 'calculate_loss',
           'SSIM', 'VGGLoss',
           'RetinexLoss', 'BlurAwareGradientLoss', 'PhaseEnhancedFrequencyLoss']
