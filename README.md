# STITCH

STITCH studies calibration between TESS observing sectors. This repository provides collection, conditional normalizing flow (CNF) and histogram gradient boosting (HGB) workflows, plus a dashboard with sector summaries and full light curves.

The frozen physical HGB is trained against **Gaia RP**, with a training-only color transfer to the TESS flux scale. The supplied historical CNF uses a **relative target**. They are distinct research models, not an architecture-only comparison. Quiet-star scatter does not establish preservation of long-duration stellar variability.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[cnf,test]'
stitch --help
pytest
```

Python 3.9 or newer is required. Frozen serialized models use the versions recorded in their bundles; scikit-learn, Torch and Zuko are pinned to the verified versions.

## Four workflows

1. **Collection:** `stitch collect --requests requests.parquet --out runs/collection`. A request lists an exact TIC, sector and archive URI; an optional existing cache path is reused. The collector downloads at most two products at once, measures SAP and PDCSAP with explicit quality cuts, and saves checksums and receipts before releasing raw bytes. It does not build a new FITS cache.
2. **CNF:** `stitch prepare --family cnf ...`, then `stitch train-cnf ...` or `stitch predict-cnf ...`. New training defaults to Gaia RP; historical relative-target reproduction is an explicit option. Frozen inference uses the saved conditional median, target scaling, strength and clipping.
3. **HGB:** `stitch prepare --family hgb ...`, then `stitch train-hgb ...` or `stitch predict-hgb ...`. Five training rounds retain the Gaia anchor. Reference and color preprocessing exclude query source groups and held-out labels.
4. **Visualization:** `stitch dashboard --predictions predictions.parquet --curves curves.parquet --out dashboard.html`. The original star grid opens sector summaries and two full-cadence panels. All-sector and individually TARS-qualified views remain separate.

The ellipses above identify workflow stages, not complete commands. See [usage](docs/usage.md) for complete examples and input schemas, [model status](docs/model_status.md) for scientific limits, and [the model manifest](models/manifest.json) for verified artifact versions. Model bundles are distributed separately from the source tree; no training dataset or bulk light-curve archive is included.

## Research conventions

Use `systematic_score > 0.95` for an individually qualified sector; selecting a quiet star does not qualify all its sectors. Report missing Gaia anchors and qualification attrition. Never normalize away a star's Gaia anchor or force separated seasons to the same measured brightness.

Historical datasets, checkpoints and research reports remain outside this core source release. The previous tracked source is retained in Git history and an archival tag. The existing test stars have been examined repeatedly and are a development benchmark, not a pristine holdout.

MIT licensed; see [LICENSE](LICENSE).
