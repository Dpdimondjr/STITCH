"""
Add Teff, logg, and radius to training_stars.csv via TIC queries.
Run once; idempotent (skips if columns already present).
"""

import pandas as pd
import numpy as np
from astroquery.mast import Catalogs
import warnings
warnings.filterwarnings("ignore")

stars = pd.read_csv("training_stars.csv")

if "Teff" in stars.columns and stars["Teff"].notna().sum() > 100:
    print("Teff already present — nothing to do.")
    raise SystemExit(0)

tic_ids = stars["tic_id"].unique().tolist()
print(f"Querying TIC for Teff/logg/rad on {len(tic_ids)} unique stars...")

CHUNK = 1000
rows = []
for i in range(0, len(tic_ids), CHUNK):
    chunk = tic_ids[i:i + CHUNK]
    try:
        cat = Catalogs.query_criteria(catalog="TIC", ID=chunk)
        if len(cat) == 0:
            continue
        cdf = cat.to_pandas()[["ID", "Teff", "logg", "rad"]].copy()
        cdf = cdf.rename(columns={"ID": "tic_id"})
        cdf["tic_id"] = pd.to_numeric(cdf["tic_id"], errors="coerce")
        rows.append(cdf)
        print(f"  [{i+len(chunk)}/{len(tic_ids)}] done")
    except Exception as e:
        print(f"  [{i}] chunk failed: {e}")

if not rows:
    print("No TIC data returned.")
    raise SystemExit(1)

teff_df = (pd.concat(rows, ignore_index=True)
             .dropna(subset=["tic_id"])
             .drop_duplicates(subset="tic_id"))
teff_df["tic_id"] = teff_df["tic_id"].astype(int)
for col in ["Teff", "logg", "rad"]:
    teff_df[col] = pd.to_numeric(teff_df[col], errors="coerce")

stars = stars.merge(teff_df, on="tic_id", how="left")
stars.to_csv("training_stars.csv", index=False)

print(f"\nDone. Teff coverage: {stars['Teff'].notna().sum()}/{len(stars)} rows")
print(f"Teff range: {stars['Teff'].min():.0f}–{stars['Teff'].max():.0f} K")
print(f"Stars with Teff > 4500K: {(stars['Teff'] > 4500).sum()}")
print(f"Stars with Teff < 4500K: {(stars['Teff'] <= 4500).sum()} (would be excluded)")
print("Saved → training_stars.csv")
