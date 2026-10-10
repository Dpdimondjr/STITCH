# Overnight Investigation Log — STITCH step-function search

Started 2026-10-08, continuing autonomously until exhausted or a real lead is found.
**NEW BEST: 0.8926% (53D: 47 base + TIC + destripe + pointing)** — confirmed real,
-0.0064pp vs 0.899%, >5x the inter-run ensemble std (0.0012%). Not a step function,
but the first genuine improvement beyond 0.899% all session. See item 1 below.

Previous best: 0.899% (49D TIC ensemble). Goal: find a genuine step-function path
toward the ~0.5-0.7% noise floor, or conclusively characterize why one isn't available
with current architecture.

## Running priority queue

1. [x] Stratified pointing/crowding check — DONE. Crowding stratification hypothesis
       REFUTED (r stays ~0.12-0.16 across numcont terciles, no concentration).
       BUT found something better: r(perr_row,dev) is almost entirely a CAM4-ONLY
       effect (Cam1=-0.012, Cam2=0.064, Cam3=0.012, Cam4=0.2015), and Cam4 is 72%
       of the matched sample, so the Oct 2 global ablation (0.4% gain) was diluted
       by pooling a real Cam4 signal against near-zero signal elsewhere. Confirmed
       non-redundant with static row feature (within-row-bin r=0.184, not weaker).
       Quantified: simple linear perr_row/perr_col fit on Cam4 raw flux_offset_loo
       gives R²=0.041, reduces raw per-star CV 1.2117%→1.1531% (4.8% relative).
       VERDICT: real, modest, same order of magnitude as contratio/numcont — not
       a step function, but a legitimate incremental feature. Launching a proper
       retrain with perr_row/perr_col added to confirm it survives in the full
       49D+destripe feature set (not just a bare linear fit).
2. [x] Re-examine z-score / signed-residual heatmaps — DONE. Extended to all 16
       cam/ccd combos using hgb_v2's actual post-correction residual (not raw).
       First pass found alarming 6.8-12% spikes in Cam3 CCD2/3, Cam4 CCD2 — but
       this was a methodology bug on my part (forgot to reapply the standard
       flux_offset_loo∈[0.85,1.15] filter + used mean not median, so a handful of
       corrupted observations, e.g. TIC 841401684/sector60 flux_offset_loo=7.84,
       dominated specific bins). Redid properly with filter+median: all 16 combos
       show sane 0.08-0.87% row/col range — consistent with earlier destripe
       diagnostic. CONCLUSION: no missed larger structure. Cam4 has the smallest
       residual range (0.08-0.17%) and lowest median|resid| (0.63-0.66%) — best
       corrected camera overall; Cam1/Cam2 CCD3/4 somewhat higher median|resid|
       (1.1-1.57%) but this doesn't contradict the earlier per-star-CV finding
       (Cam1 best, Cam4 worst) since it's a different statistic (absolute per-
       observation deviation vs per-star cross-sector consistency after LOO
       normalization already removes constant per-star bias).
3. [x] Failure case analysis — DONE. Worst 1% (n=799, |resid|>=6.77%) vs best 50%:
       - contratio 3.9x higher (0.0717 vs 0.0184) — STRONGEST signal found tonight
       - cdpp1_0 1.59x higher (intrinsically noisier target stars)
       - numcont 1.20x higher, n_sectors_total 0.75x (fewer sectors)
       - Camera skew: worst 1% is 67.5% Cam4 vs 76.5% in good population (Cam4
         underrepresented in failures despite being largest population — consistent
         with Cam4 being best-corrected per item 2)
       - Sectors 24, 25 massively overrepresented (2.85x, 3.25x) — matches the
         sparse/high-variance sectors flagged in the earlier idea-5 investigation
       - Sectors 18, 19, 4 also overrepresented (1.9-2.1x) — matches the TESS-wide
         sector-17-19 event and sector-3-5 cluster found earlier
       Full-population check: r(contratio,|resid|)=0.20, r(log1p(contratio),|resid|)
       =0.24 (STRONGER than raw) — relationship is flat below ~3% contratio then
       accelerates sharply above it (deciles 6-9: 0.0085→0.0091→0.0110→0.0154).
       Tested contratio×pointing interaction in Cam4: REFUTED — r(perr_row,dev)
       actually DECREASES with higher contratio tercile (0.236→0.206→0.171).
       Pointing and crowding are independent mechanisms, not synergistic.
       ACTIONABLE: log1p(contratio) as an explicit feature (r=0.24 vs 0.20 raw)
       is a cheap, well-motivated addition — queued for next retrain.
4. [x] Per-cadence feature feasibility pilot — DONE. 80/80 downloaded successfully
       (40 worst-residual + 40 random, one sector each). Results:
         r(trend_slope, |resid|)   = 0.070
         r(max_jump, |resid|)      = 0.157
         r(outlier_frac, |resid|)  = 0.164
       HONEST CAVEAT: n=80 with a stratified (not purely random) sample gives
       SE≈0.11-0.22 on these correlations — suggestive, not statistically
       confirmed at this sample size. max_jump and outlier_frac are plausible
       (roughly numcont-sized effect, r~0.16-0.18) but need a properly random,
       larger sample (200+) before trusting as a real feature. NOT validated
       tonight — a promising lead for a dedicated follow-up, not a confirmed win.
5. [ ] Targeted centroid feature build (conditional on #1 showing signal).
6. [ ] Crowding-weighted KNN (weight neighbors by crowding similarity, not just distance).
7. [x] Final tabular bundle: DONE. destripe + pointing + log1p(contratio), 54D.
       Result: 0.8926% — EXACTLY identical to destripe+pointing alone (+0.0000pp).
       log1p(contratio) adds nothing: HGB trees (max_leaf_nodes=1023) already
       approximate the nonlinear contratio→error relationship via splits on the
       raw feature; a pre-transformed feature is redundant for tree models
       specifically (would not necessarily generalize to linear/NN architectures).
       CONFIRMED FINAL OVERNIGHT BEST: 0.8926% (47 base + TIC + destripe +
       pointing, 53D). bp_rp still blocked on the known column-name bug (not
       fixed tonight — out of scope for this session).
8. [ ] NEW — sample-weight by contratio during HGB training. Discovered the HGB
       ensemble pipeline has ZERO sample weighting (only the CNF scripts have
       oracle-ceiling weights) — given contratio is now the strongest known error
       predictor (r=0.20-0.24, nonlinear, failure-analysis-confirmed 3.9x
       enrichment in worst 1%), upweighting high-contratio observations during
       training could directly target the population driving the worst errors.
       Cheap to test (just add sample_weight to the HGB.fit() calls) — queued for
       after current jobs finish to avoid resource contention.
9. [ ] Documentation fix (not a lever, just accuracy): contratio is MAST's
       contaminant/target flux RATIO (can exceed 1.0 — 0.36% of stars do, max
       observed 19.17), not a bounded "fraction of aperture flux" as currently
       described in the progress report. Small wording fix needed there.

## Confirmed dead ends (full list, do not re-test)

- Tabular feature ceiling ~0.904% (Oct 2026) — no feature combo exceeds it
- Spatial polynomial field — overfits, closed
- Sector-level mean bias correction — redundant with existing `sector_mean_loo` feature
  (confirmed numerically for sectors 3-5, 17-19 — matches raw bias almost exactly)
- Test-time per-star recalibration — backfires (+9.5% worse) due to LOO anti-correlation
- Normalizing flow (any variant, including 49D TIC features) — always worse than HGB
  (CNF+49D = 1.017% vs HGB 47D = 0.944%, HGB ensemble 49D = 0.899%)
- Corner-specific effects — refuted, no consistent pattern across cameras
- Quiet-catalog per-sector contamination — refuted (paired test: unverified sectors
  show LOWER not higher CV, 1.251% vs 1.315%)
- Data volume / collection density as bottleneck — refuted (Cam1, worst KNN radius
  582px, has BEST final CV 0.691%; Cam4, best radius 119px, has WORST 0.893% —
  dominated by selection-order sampling bias in collect_new_stars.py, not density)
- LOO-label sampling noise as dominant driver — refuted (CV×sqrt(N-1) climbs 3.3x
  with N instead of staying flat, meaning real signal grows with N, not noise shrinking)
- Calendar baseline / multi-camera switching as explanation for N-CV relationship —
  refuted (r≈0, non-monotonic bins)

## Real findings this session (small but validated)

- Destriping (1D marginal column/row features): real, reproduces across independent
  splits, confirmed present in post-correction residual — but invisible to the global
  median metric (affects ~5% of stars on a fraction of their sectors). Kept in model,
  no harm, no measurable global gain (0.8990% vs 0.899%).
- Ecliptic crowding mechanism for Cam3: confirmed (r=0.042 with numcont, p<0.001)
- contratio/numcont catalog features: real modest gain, 0.944% → 0.899% in ensemble
- Pointing (FITS centroids, perr_col/perr_row): real but small GLOBAL gain
  (0.9040% → 0.9001%, ~0.4% relative) — untested whether this concentrates in
  high-crowding subpopulation (queue item #1)
- gaiarp is 99.9% redundant with tmag (r=0.9994) — bp_rp is the only real unexploited
  color info; still blocked on the known column-name bug in fetch_tic_contamination.py
- numcont correlates strongly (non-linearly) with n_sectors_total: median rises
  60→64→78→82→95→100 across N bins despite near-zero linear r — CVZ-like high-N
  stars are disproportionately crowded. This is the leading explanation for why
  CV×sqrt(N-1) grows with N (not measurement noise, not calendar drift — crowding).

## Session closed out — summary

Progress report updated (v10) with all findings above. Models saved:
- models/hgb_destripe_ensemble/ (0.8990%, destripe only — null on global metric)
- models/hgb_destripe_pointing_ensemble/ (0.8926%, destripe+pointing — REAL improvement)
- models/hgb_final_combined_ensemble/ (0.8926%, +log1p(contratio) — no additional gain)

New best going forward: 0.893% (was 0.899%). Modest but real; not a step function.
The step-function question remains open — nothing tested tonight broke through the
architecture ceiling. Per-cadence features are the most promising untested lead
(r=0.16-0.18 in a small pilot) but need a proper larger sample before trusting.

## Architectural step-function candidates (unchanged, still the fallback)

- Per-cadence time-domain features (moderate effort/risk)
- Focal-plane field model, CNN/ViT/GP over FFI pixels (high effort, true step function)

## Round 2 — "try them" (2026-10-09, user asked to actually test both candidates)

### Per-cadence time-domain features, v2 (proper random sample, n=300) — DONE, VALIDATED
v1 (overnight) used a 50/50 worst/random stratified n=80 sample — SE too large
to trust. Relaunched with a PURE random sample, n=300, 100% download success.

Raw correlations with |residual|:
  trend_slope:   r=-0.1735  SIGNIFICANT
  outlier_frac:  r=+0.1428  SIGNIFICANT
  max_jump:      r=-0.0252  not significant (the v1 "finding" of r=0.157 for
                 this feature did NOT replicate — confirms it was an artifact
                 of the small biased sample, correctly flagged as unconfirmed)
  var_ratio:     r=-0.0308  not significant

Redundancy check against existing scalars (jitter_rms + pdcvar, which have
100% coverage vs cdpp1_0's 14.2% — had to drop cdpp1_0 from the control set
after an initial run collapsed to n=21 and gave a misleading huge-but-noisy
partial correlation):
  trend_slope:   partial_r=-0.1755  SIGNIFICANT (unchanged from raw — NOT
                 explained by jitter_rms/pdcvar at all)
  outlier_frac:  partial_r=+0.1560  SIGNIFICANT (slightly increased — also
                 not redundant)

CONCLUSION: both features are real, validated, non-redundant at proper sample
size. This is a genuine, confirmed lever — unlike nearly everything else
tested across both sessions tonight.

### Pixel-level focal-plane MVP — eval/pixel_level_pilot.py — DONE, promising
First real step beyond aperture-summed tabular features: downloaded actual
TESS-SPOC Target Pixel Files (11x11 cutouts, native cadence resolution) for
120 pure-random test stars (100% download success). Extracted 4 features
directly from pixels: centroid_offset, centroid_jitter (per-cadence pointing
stability measured from pixels, not the pipeline's aggregate), asymmetry
(quadrupole moment imbalance), secondary_peak_frac (brightest pixel outside
the central 5x5 PSF core, relative to the core peak — detects a resolvable
neighbor source).

Raw correlations with |residual| (n=120): all NOT significant at 95%
  (centroid_offset r=0.090, centroid_jitter r=0.120, asymmetry r=0.088,
   secondary_peak_frac r=0.144)

PARTIAL correlations controlling for contratio+numcont — the real test:
  centroid_jitter:      partial_r=+0.211  SIGNIFICANT
  secondary_peak_frac:  partial_r=+0.198  SIGNIFICANT
  centroid_offset:      partial_r=+0.066  not significant
  asymmetry:            partial_r=+0.062  not significant

INTERPRETATION: raw correlations were suppressed because pixel shape and
contratio are collinear (both reflect crowding) — but once you remove what
contratio/numcont already explain, centroid_jitter and secondary_peak_frac
still carry real, independent signal. This is the strongest evidence yet for
the focal-plane/pixel-level approach: it's the only result from either "try
them" test that shows genuinely NEW, non-redundant information rather than a
different encoding of something already in the model.

CAVEAT: n=120 is still modest for a partial-correlation test (2 extra DOF
consumed by the regression). Worth a larger confirmatory run (n=300+) before
committing to full-scale feature extraction (which requires TPF downloads for
the entire train/val/test population — a much bigger undertaking, thousands
of ~3MB files).

NEXT STEP IF CONFIRMED: build centroid_jitter + secondary_peak_frac as actual
training features (requires scaling TPF extraction to ~37K train + 4.7K val +
4.7K test stars — non-trivial storage/compute, should be scoped as its own
project rather than squeezed into this session).
