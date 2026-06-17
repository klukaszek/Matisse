"""Parity tests for a manual, MPS-native bilinear grid sampler.

Goal: replace F.grid_sample in the gradient path (P_cell_position.efficient_warping)
so the backward pass stays on-device instead of falling back to CPU via the
unimplemented aten::grid_sampler_2d_backward on MPS.
"""
import torch
import torch.nn.functional as F


def bilinear_grid_sample(input, grid, align_corners=True):
    """Bilinear F.grid_sample equivalent (padding_mode='zeros') from ops with
    full MPS forward+backward support (floor/clamp/gather/mul/add).

    input: (N, C, H, W)   grid: (N, Ho, Wo, 2), last dim = (x, y) in [-1, 1]
    """
    N, C, H, W = input.shape
    _, Ho, Wo, _ = grid.shape

    x = grid[..., 0]
    y = grid[..., 1]

    if align_corners:
        ix = (x + 1) * 0.5 * (W - 1)
        iy = (y + 1) * 0.5 * (H - 1)
    else:
        ix = ((x + 1) * W - 1) * 0.5
        iy = ((y + 1) * H - 1) * 0.5

    ix0 = torch.floor(ix)
    iy0 = torch.floor(iy)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    wx1 = ix - ix0
    wx0 = 1 - wx1
    wy1 = iy - iy0
    wy0 = 1 - wy1

    w00 = (wx0 * wy0).unsqueeze(1)
    w01 = (wx1 * wy0).unsqueeze(1)
    w10 = (wx0 * wy1).unsqueeze(1)
    w11 = (wx1 * wy1).unsqueeze(1)

    flat = input.reshape(N, C, H * W)

    def corner(ixc, iyc):
        in_bounds = (ixc >= 0) & (ixc <= W - 1) & (iyc >= 0) & (iyc <= H - 1)
        ixc = ixc.clamp(0, W - 1).long()
        iyc = iyc.clamp(0, H - 1).long()
        idx = (iyc * W + ixc).reshape(N, 1, Ho * Wo).expand(N, C, -1)
        vals = torch.gather(flat, 2, idx).reshape(N, C, Ho, Wo)
        return vals * in_bounds.unsqueeze(1).to(vals.dtype)

    out = (corner(ix0, iy0) * w00 + corner(ix1, iy0) * w01
           + corner(ix0, iy1) * w10 + corner(ix1, iy1) * w11)
    return out


def _run(device):
    torch.manual_seed(0)
    N, C, H, W, Ho, Wo = 2, 3, 17, 19, 23, 21
    inp = torch.randn(N, C, H, W, device=device, requires_grad=True)
    # grid spanning slightly past [-1, 1] so the zeros-padding path is exercised
    grid = (torch.rand(N, Ho, Wo, 2, device=device) * 2.4 - 1.2).requires_grad_(True)

    ref = F.grid_sample(inp, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    mine = bilinear_grid_sample(inp, grid, align_corners=True)
    fwd_err = (ref - mine).abs().max().item()

    go = torch.randn_like(ref)
    gi_ref, gg_ref = torch.autograd.grad(ref, [inp, grid], go, retain_graph=True)
    gi_mine, gg_mine = torch.autograd.grad(mine, [inp, grid], go)
    din_err = (gi_ref - gi_mine).abs().max().item()
    dgrid_err = (gg_ref - gg_mine).abs().max().item()

    print(f'[{device}] fwd={fwd_err:.2e} d_input={din_err:.2e} d_grid={dgrid_err:.2e}')
    return fwd_err, din_err, dgrid_err


if __name__ == '__main__':
    print('manual sampler vs F.grid_sample (bilinear, zeros, align_corners=True)')
    _run('cpu')
    if torch.backends.mps.is_available():
        # cross-device: manual on MPS vs reference on CPU
        torch.manual_seed(0)
        N, C, H, W, Ho, Wo = 2, 3, 17, 19, 23, 21
        inp = torch.randn(N, C, H, W, requires_grad=True)
        grid = (torch.rand(N, Ho, Wo, 2) * 2.4 - 1.2).requires_grad_(True)
        ref = F.grid_sample(inp, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        go = torch.randn_like(ref)
        gi_ref, gg_ref = torch.autograd.grad(ref, [inp, grid], go)

        inp_m = inp.detach().to('mps').requires_grad_(True)
        grid_m = grid.detach().to('mps').requires_grad_(True)
        mine = bilinear_grid_sample(inp_m, grid_m, align_corners=True)
        gi_m, gg_m = torch.autograd.grad(mine, [inp_m, grid_m], go.to('mps'))
        print(f'[mps-vs-cpu] fwd={(ref - mine.cpu()).abs().max():.2e} '
              f'd_input={(gi_ref - gi_m.cpu()).abs().max():.2e} '
              f'd_grid={(gg_ref - gg_m.cpu()).abs().max():.2e}')
