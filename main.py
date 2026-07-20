import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

tic_id = "TIC 264221449"

search_result = lk.search_lightcurve(tic_id, mission="TESS", author="SPOC")
print(search_result)

lc_collection = search_result.download_all()

sap_lcs = []
pdcsap_lcs = []

for lc in lc_collection:
    sap_lcs.append(lc.select_flux("sap_flux"))
    pdcsap_lcs.append(lc.select_flux("pdcsap_flux"))

# Stitch without per-sector normalization
sap_stitched = lk.LightCurveCollection(sap_lcs).stitch(corrector_func=None)
pdcsap_stitched = lk.LightCurveCollection(pdcsap_lcs).stitch(corrector_func=None)

# Normalize once globally
sap_stitched = sap_stitched / np.nanmedian(sap_stitched.flux.value)
pdcsap_stitched = pdcsap_stitched / np.nanmedian(pdcsap_stitched.flux.value)

# Plot
# fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
# fig.suptitle(f"{tic_id} - Sector Discontinuities", fontsize=14)

# axes[0].plot(sap_stitched.time.value, sap_stitched.flux.value, 'k.', markersize=0.5, alpha=0.5)
# axes[0].set_ylabel("SAP FLUX (normalized)")

# axes[1].plot(pdcsap_stitched.time.value, pdcsap_stitched.flux.value, 'k.', markersize=0.5, alpha=0.5)
# axes[1].set_ylabel("PDCSAP FLUX (normalized)")
# axes[1].set_xlabel("Time (BTJD)")

# Mark sector boundaries
# for lc in sap_lcs:
#     t_start = lc.time.value[0]
#     axes[0].axvline(t_start, color='red', alpha=0.3, linewidth=0.8)
#     axes[1].axvline(t_start, color='red', alpha=0.3, linewidth=0.8)

# plt.tight_layout()
# plt.savefig("sector_discontinuities.png", dpi=150)
# plt.show()

# Per-sector median flux
print("\nPer-sector median flux (PDCSAP globally normalized):")
for i, lc in enumerate(pdcsap_lcs):
    # Apply global normalization factor
    global_median = np.nanmedian(np.concatenate([l.flux.value for l in pdcsap_lcs]))
    median = np.nanmedian(lc.flux.value) / global_median
    sector = lc.meta.get('SECTOR', i)
    print(f"  Sector {sector}: median={median:.4f}")

print("\nDetector metadata per sector:")
for lc in sap_lcs:
    sector = lc.meta.get('SECTOR', '?')
    camera = lc.meta.get('CAMERA', '?')
    ccd = lc.meta.get('CCD', '?')
    print(f"\nSector {sector} (Camera {camera}, CCD {ccd}):")
    for key, value in lc.meta.items():
        print(f"  {key}: {value}")

records = []
global_median = np.nanmedian(np.concatenate([l.flux.value for l in pdcsap_lcs]))

for sap_lc, pdcsap_lc in zip(sap_lcs, pdcsap_lcs):
    sector_median = np.nanmedian(pdcsap_lc.flux.value) / global_median
    
    records.append({
        'sector': sap_lc.meta.get('SECTOR'),
        'camera': sap_lc.meta.get('CAMERA'),
        'ccd': sap_lc.meta.get('CCD'),
        'pxtable': sap_lc.meta.get('PXTABLE'),
        'crowdsap': sap_lc.meta.get('CROWDSAP'),
        'flfrcsap': sap_lc.meta.get('FLFRCSAP'),
        'meanblca': sap_lc.meta.get('MEANBLCA'),
        'meanblcb': sap_lc.meta.get('MEANBLCB'),
        'meanblcc': sap_lc.meta.get('MEANBLCC'),
        'meanblcd': sap_lc.meta.get('MEANBLCD'),
        'pdcvar': sap_lc.meta.get('PDCVAR'),
        'cdpp1_0': sap_lc.meta.get('CDPP1_0'),
        'flux_offset': sector_median
    })

df = pd.DataFrame(records).sort_values('sector')
print(df.to_string(index=False))

# Plot flux offset vs key features
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Flux Offset vs Detector Metadata", fontsize=14)

features = ['pxtable', 'crowdsap', 'flfrcsap', 'meanblca', 'pdcvar', 'cdpp1_0']
for ax, feat in zip(axes.flatten(), features):
    ax.scatter(df[feat], df['flux_offset'], c=df['camera'], cmap='tab10', s=100)
    ax.set_xlabel(feat)
    ax.set_ylabel('flux offset (normalized)')
    ax.set_title(feat)
    # Label each point with sector number
    for _, row in df.iterrows():
        ax.annotate(str(int(row['sector'])), 
                   (row[feat], row['flux_offset']),
                   textcoords="offset points", xytext=(5,5), fontsize=7)

plt.tight_layout()
plt.savefig("metadata_vs_offset.png", dpi=150)
plt.show()