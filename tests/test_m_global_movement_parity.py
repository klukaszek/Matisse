"""Parity test for the M_global_movement vectorization.

`reference_forward` is a verbatim copy of the ORIGINAL python-loop algorithm.
The test asserts the (possibly refactored) live DefaultGlobalMovement.forward
produces a bit-identical pred_dxy. pred_dxy is derived from integer shifts, so
on decisive (non-tie) inputs the match must be exact.
"""
import os
import sys
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import numpy as np
import torch
import torch.nn.functional as F

from root_config import ROOT_DIR, DEVICE
from Experiment.helper import compute_required_image_resolution
from Simulated.Cortex.M_global_movement.M_Default import DefaultGlobalMovement
from Simulated.Cortex.P_cell_position.P_Default import DefaultCellPosition


def reference_forward(self, ons1, ons2, P_cell_position, true_dxy=None):
    with torch.no_grad():
        ons1_ = ons1.detach().clone()
        ons2_ = ons2.detach().clone()
        blurred_ons1 = self.gaussian_blur(ons1_)
        blurred_ons2 = self.gaussian_blur(ons2_)
        xy_full = P_cell_position.get_XY_default_locations()
        required_image_resolution = compute_required_image_resolution(xy_full.detach().clone())
        grid = self.generate_grid_fixed(xy_full[0, :, 0, 0], xy_full[0, :, -1, 0], xy_full[0, :, 0, -1], xy_full[0, :, -1, -1], required_image_resolution)
        uvs = P_cell_position.get_UV_locations(grid.permute(2, 0, 1).unsqueeze(0))
        uvs = uvs.repeat(len(blurred_ons1), 1, 1, 1)
        full_blurred_ons1 = F.grid_sample(blurred_ons1, uvs.permute(0, 2, 3, 1), align_corners=True, mode='bilinear', padding_mode='zeros')
        full_blurred_ons2 = F.grid_sample(blurred_ons2, uvs.permute(0, 2, 3, 1), align_corners=True, mode='bilinear', padding_mode='zeros')
        mask = torch.ones_like(blurred_ons1).to(self.device)
        full_mask = F.grid_sample(mask, uvs.permute(0, 2, 3, 1), align_corners=True, mode='nearest', padding_mode='zeros')
        levels = int(np.log2(required_image_resolution)) - 2
        pyramid_blurred_ons1 = [full_blurred_ons1]
        pyramid_blurred_ons2 = [full_blurred_ons2]
        pyramid_mask = [full_mask]
        for i in range(levels):
            pyramid_blurred_ons1.append(F.avg_pool2d(pyramid_blurred_ons1[-1], 2, 2))
            pyramid_blurred_ons2.append(F.avg_pool2d(pyramid_blurred_ons2[-1], 2, 2))
            pyramid_mask.append(F.max_pool2d(pyramid_mask[-1], 2, 2))
        pyramid_blurred_ons1 = pyramid_blurred_ons1[::-1]
        pyramid_blurred_ons2 = pyramid_blurred_ons2[::-1]
        pyramid_mask = pyramid_mask[::-1]
        aggregate_shifts = torch.zeros(len(full_blurred_ons1), 2, device=self.device)
        for i in range(levels + 1):
            current_blurred_ons1 = pyramid_blurred_ons1[i]
            current_blurred_ons2 = pyramid_blurred_ons2[i]
            current_mask = pyramid_mask[i]
            current_res = current_blurred_ons1.shape[2]
            padded_current_blurred_ons1 = F.pad(current_blurred_ons1, (2, 2, 2, 2), mode='constant', value=0)
            padded_current_mask = F.pad(current_mask, (2, 2, 2, 2), mode='constant', value=0)
            shifted_ons1_list = [padded_current_blurred_ons1[:, :, dx + 2:dx + 2 + current_res, dy + 2:dy + 2 + current_res] for dx, dy in self.shifts]
            shifted_mask_list = [padded_current_mask[:, :, dx + 2:dx + 2 + current_res, dy + 2:dy + 2 + current_res] for dx, dy in self.shifts]
            ncc_scores = torch.ones(len(current_blurred_ons1), len(self.shifts), device=self.device) * 1000
            for j, (shifted_ons1, shifted_mask) in enumerate(zip(shifted_ons1_list, shifted_mask_list)):
                error = torch.sum(torch.sqrt((shifted_ons1 - current_blurred_ons2) ** 2) * (shifted_mask * current_mask), dim=(2, 3)) / (torch.sum(shifted_mask * current_mask, dim=(2, 3)) + 1e-6)
                ncc_scores[:, j] = error.squeeze()
            _, max_indices = torch.min(ncc_scores, 1)
            dx = max_indices % len(self.shift_range) - 2
            dy = max_indices // len(self.shift_range) - 2
            aggregate_shifts += torch.stack([dx, dy], -1) * (2 ** (levels - i))
            for j in range(i + 1, len(pyramid_blurred_ons1)):
                cdx = dy * 2 ** (j - i)
                cdy = dx * 2 ** (j - i)
                P = 2 ** (j - i) * 2
                new_blurred_ons1 = F.pad(pyramid_blurred_ons1[j], (P, P, P, P), mode='constant', value=0)
                current_res = pyramid_blurred_ons1[j].shape[2]
                new_blurred_ons1 = [new_blurred_ons1[k, :, cdx[k] + P:cdx[k] + P + current_res, cdy[k] + P:cdy[k] + P + current_res] for k in range(len(new_blurred_ons1))]
                pyramid_blurred_ons1[j] = torch.stack(new_blurred_ons1)
                new_mask = F.pad(pyramid_mask[j], (P, P, P, P), mode='constant', value=0)
                new_mask = [new_mask[k, :, cdx[k] + P:cdx[k] + P + current_res, cdy[k] + P:cdy[k] + P + current_res] for k in range(len(new_mask))]
                pyramid_mask[j] = torch.stack(new_mask)
        pred_dxy = aggregate_shifts.reshape([-1, 2]) / (required_image_resolution / 2)
        pred_dxy = pred_dxy.detach().unsqueeze(1)
    return pred_dxy


with open(f'{ROOT_DIR}/Experiment/Config/Default/LMS.yaml') as f:
    params = yaml.safe_load(f)

torch.manual_seed(0)
np.random.seed(0)
M = DefaultGlobalMovement(params, DEVICE)
P = DefaultCellPosition(params, DEVICE)

BS = params['Dataset']['batch_size']
sim = params['Experiment']['simulation_size']

# Decisive inputs: ons2 is an integer-shifted, slightly-noised ons1 so the
# coarse-to-fine NCC has clear (non-tie) minima -> stable argmin.
base = torch.rand(BS, 1, sim, sim, device=DEVICE)
base = F.avg_pool2d(F.pad(base, (4, 4, 4, 4), mode='reflect'), 3, 1)  # smooth a bit
base = base[:, :, :sim, :sim]
ons1 = base
ons2 = torch.roll(base, shifts=(3, 2), dims=(2, 3)) + 0.01 * torch.rand_like(base)

ref = reference_forward(M, ons1, ons2, P)
new = M.forward(ons1, ons2, P, None)

max_diff = (ref - new).abs().max().item()
exact = torch.equal(ref, new)
print(f'[{DEVICE}] pred_dxy shape={tuple(new.shape)} max_abs_diff={max_diff:.3e} exact={exact}')
print('unique ref dxy:', torch.unique(ref).tolist()[:10])
assert exact, f'pred_dxy mismatch (max abs diff {max_diff})'
print('PASS: vectorized M_global_movement matches reference exactly')
