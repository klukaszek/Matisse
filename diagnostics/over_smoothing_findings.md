# Over-smoothing localization — diagnostic findings (Jul 2026)

Goal: decide whether the LPIPS/perceptual regression seen when moving from `LMS_L1`
to `LMS_L1_Fixational` is caused by the eye motion, the decoder, the retinal noise,
or honest peripheral sampling loss — so we know what belongs in Matisse vs Retinax.

Script: `scratchpad/diag.py` (2-factor over trained 100k checkpoints + noise toggle
+ RAPSD + eccentricity). Figure: `diagnostics/rapsd_2factor.png`.

## 2-factor design (existing checkpoints)
- **A** = `LMS_L1`               — Default decoder, uniform motion
- **B** = `LMS_L1_Implicit_Gaussian` — Implicit-Gaussian decoder, uniform motion
- **C** = `LMS_L1_Fixational`    — Implicit-Gaussian decoder, fixational motion

B−A isolates the **decoder**; C−B isolates the **eye motion**. (The user's original
A-vs-C comparison changed BOTH → confounded.)

## Results (noise ON, mean over test set)
| | PSNR | SSIM | LPIPS | gradE recon/target |
|---|---|---|---|---|
| A Default dec, uniform      | 22.13 | 0.665 | 0.306 | 0.567 |
| B Implicit dec, uniform     | 21.59 | 0.619 | 0.430 | 0.558 |
| C Implicit dec, fixational  | 22.36 | 0.653 | 0.390 | 0.421 |
| C, NOISE OFF                | 22.44 | 0.660 | 0.374 | 0.415 |

LPIPS decomposes ~additively: C−A (+0.084) = **decoder +0.124** + **motion −0.040**.

## Conclusions
1. **The perceptual regression is DECODER-bound, not eye-motion-bound.** The
   Implicit-Gaussian decoder (`num_frequencies=2`, `gaussian_sigma=2`) is a large
   LPIPS/SSIM regression vs the Default decoder. Its penalty is **artifacts**, not
   blur (HF retention ~ equal to Default; grad-energy ~ equal at B).
2. **Fixational eye motion HELPS** (C better than B on LPIPS/SSIM/PSNR). It smooths
   away the Implicit decoder's HF artifacts (grad-energy drops B→C but LPIPS
   improves). The fixational work is validated — keep it.
3. **Retinal noise is a MINOR contributor** (~4% of the LPIPS gap; 0.390→0.374 off).
   The current noise model is Poisson-*form* (variance ∝ mean, ≤~1%, random global
   `ratio`) — biologically it should be **sub-Poisson** (Fano<1, refractory, history-
   dependent) — but it is NOT what's limiting reconstruction now.
4. **Error is location-independent** (contrast-normalized periph/fovea = 1.23×) →
   decoder/re-encoder-bound, NOT peripheral-sampling-bound. Matches the reviewer's
   original "edge-everywhere" read.

## Where the work belongs
### Matisse (cheap, rapid — the current bottleneck is cortical)
- Decoder is the lever. Cheap retrains: revert to Default decoder, OR raise the
  Implicit decoder's bandwidth (`num_frequencies` 2→6/10, lower `gaussian_sigma`),
  re-measure LPIPS. This is a config change, plays to Matisse's strengths.
- Keep fixational eye motion (it helps at matched decoder).

### Retinax (the biology that Matisse structurally can't host)
- `SpikeConversion` is currently the **identity** — there is no RGC output code at
  all. Sub-Poisson, refractory, history-dependent, temporally-precise spiking is the
  biologically-correct output model AND the substrate for drift→temporal-integration
  →hyperacuity (Intoy & Rucci 2020). This needs temporal dynamics Matisse's static
  per-frame pipeline lacks → build it in Retinax.
- Diagnostics say the retina/sampling/noise are NOT today's reconstruction limiter,
  so the Retinax spike-code work is about **biological accuracy + future hyperacuity**,
  not fixing the current blur (which is cortical/decoder).
- Priors: the cortical loss is L1 in the whitened ONS domain (reasonable). If Retinax
  wants sharpness to *emerge*, the thesis-consistent lever is an efficient-coding /
  natural-image prior (sparse gradients), NOT a trained perceptual (LPIPS) loss —
  LPIPS stays a diagnostic only.
