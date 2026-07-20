#!/bin/bash
# Consolidates CSVs into training_data.parquet every 30 minutes.
# Run with: nohup bash consolidate_loop.sh > /tmp/consolidate_loop.log 2>&1 &

cd /Users/daviddimond/Documents/STITCH

while true; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo "[$(date)] Starting consolidation..."

    # Backup current parquet
    cp training_data.parquet "training_data_backup_${TIMESTAMP}.parquet" \
        && echo "[$(date)] Backup → training_data_backup_${TIMESTAMP}.parquet" \
        || echo "[$(date)] WARNING: backup failed"

    # Consolidate
    python3 - <<'EOF'
import os, pandas as pd

STAR_CACHE_DIR = "./tess_cache/star_records_v2"
OLD_CACHE_DIR  = "./tess_cache/star_records"
OUT_FILE       = "training_data.parquet"

frames = []
for d in [OLD_CACHE_DIR, STAR_CACHE_DIR]:
    if not os.path.isdir(d):
        continue
    for f in os.listdir(d):
        if f.startswith("tic_") and f.endswith(".csv"):
            try:
                frames.append(pd.read_csv(os.path.join(d, f)))
            except Exception:
                pass

if not frames:
    print("No CSVs found — skipping write")
else:
    final = (pd.concat(frames, ignore_index=True)
               .drop_duplicates(subset=["tic_id", "sector"])
               .sort_values(["tic_id", "sector"])
               .reset_index(drop=True))
    final.to_parquet(OUT_FILE + ".tmp", index=False)
    os.replace(OUT_FILE + ".tmp", OUT_FILE)
    print(f"  {len(final):,} records, {final['tic_id'].nunique():,} stars → {OUT_FILE}")
EOF

    # Keep only the 3 most recent backups to avoid filling disk
    ls -t training_data_backup_*.parquet 2>/dev/null | tail -n +4 | xargs rm -f

    # Clean MAST download cache — FITS files are only needed during processing;
    # the extracted data is already saved in per-star CSVs.
    # Use find -delete to avoid permission issues with actively-written subdirs.
    MAST_DIR="./tess_cache/mastDownload"
    if [ -d "$MAST_DIR" ]; then
        BEFORE=$(du -sm "$MAST_DIR" 2>/dev/null | cut -f1)
        find "$MAST_DIR" -name "*.fits" -delete 2>/dev/null
        find "$MAST_DIR" -mindepth 1 -empty -delete 2>/dev/null
        echo "[$(date)] Cleaned mastDownload (freed ~${BEFORE} MB)"
    fi

    # Warn if disk is getting tight
    AVAIL=$(df -m . | awk 'NR==2 {print $4}')
    echo "[$(date)] Disk available: ${AVAIL} MB"
    if [ "$AVAIL" -lt 20480 ]; then
        echo "[$(date)] WARNING: less than 20 GB free — consider pausing collection"
    fi

    echo "[$(date)] Done. Sleeping 30 min..."
    sleep 1800
done
