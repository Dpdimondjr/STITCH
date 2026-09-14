"""
Download FFI images for all cam×ccd combinations, sectors S_LO–S_HI.
Uses HTTP Range requests to fetch only the FITS header + first PARTIAL_ROWS
rows of image data (~4MB per file instead of ~33MB).

Usage:
  python3 eval/download_ffi_pngs.py [s_lo] [s_hi]
  python3 eval/download_ffi_pngs.py 1 30
"""
import math, os, re, sys, warnings
warnings.filterwarnings("ignore")
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

S_LO        = int(sys.argv[1]) if len(sys.argv) > 1 else 1
S_HI        = int(sys.argv[2]) if len(sys.argv) > 2 else 30
OUTDIR      = "eval/ffi_pngs"
MAST        = "https://archive.stsci.edu/missions/tess/ffi"
PARTIAL_ROWS = 2048   # how many rows to keep (2048 = full image)
MAX_HDR_FETCH = 65536  # 64 KB – enough to read any FITS primary header

os.makedirs(OUTDIR, exist_ok=True)

# Cam 4 first (user priority), then cam 1-3
CAM_ORDER = [4, 1, 2, 3]
CCDS = [1, 2, 3, 4]


def find_ffi_url(sector, cam, ccd):
    try:
        r = requests.get(f"{MAST}/s{sector:04d}/", timeout=20)
        years = re.findall(r'href="(\d{4}/)"', r.text)
        target = f"{cam}-{ccd}/"
        for year in years:
            r2 = requests.get(f"{MAST}/s{sector:04d}/{year}", timeout=15)
            for doy in re.findall(r'href="(\d{3}/)"', r2.text):
                base = f"{MAST}/s{sector:04d}/{year}{doy}"
                r3 = requests.get(base, timeout=15)
                if target not in re.findall(r'href="(\d+-\d+/)"', r3.text):
                    continue
                r4 = requests.get(base + target, timeout=15)
                files = re.findall(r'href="(tess[^"]+_ffic\.fits)"', r4.text)
                if files:
                    return base + target + files[0]
    except Exception as e:
        print(f"    URL search error: {e}")
    return None


def parse_fits_header(raw_bytes):
    """Return (header_size_bytes, naxis1, naxis2, bitpix) from raw FITS bytes."""
    hdr_size = None
    for block_start in range(0, len(raw_bytes), 2880):
        block = raw_bytes[block_start: block_start + 2880]
        for rec_start in range(0, len(block), 80):
            rec = block[rec_start: rec_start + 80]
            if rec[:3] == b'END':
                hdr_size = block_start + 2880
                break
        if hdr_size:
            break
    if hdr_size is None:
        hdr_size = 2880 * 3  # fallback

    hdr_str = raw_bytes[:hdr_size].decode("ascii", errors="ignore")
    def pick(key):
        m = re.search(rf'{key}\s*=\s*(-?\d+)', hdr_str)
        return int(m.group(1)) if m else None

    naxis1 = pick("NAXIS1") or 2048
    naxis2 = pick("NAXIS2") or 2048
    bitpix = pick("BITPIX") or -32
    return hdr_size, naxis1, naxis2, bitpix


def patch_naxis2(raw_bytes, new_naxis2):
    """Overwrite the NAXIS2 value in the in-memory FITS header bytes."""
    raw = bytearray(raw_bytes)
    key = b'NAXIS2  ='
    idx = raw.find(key)
    if idx == -1:
        return bytes(raw)
    # FITS card is 80 bytes; value field is cols 10-79
    card_end = idx + 80
    card = raw[idx:card_end].decode("ascii")
    # Replace value (integer in fixed-format field)
    new_card = f"NAXIS2  = {new_naxis2:>20d}{card[30:]}"
    raw[idx:card_end] = new_card[:80].encode("ascii")
    return bytes(raw)


def download_partial(url, n_rows=PARTIAL_ROWS):
    """
    Fetch header + n_rows of image data via Range requests.
    Returns (image_2d_array, naxis1_original, naxis2_original) or None on failure.
    """
    import tempfile

    # Step 1: fetch header
    try:
        r = requests.get(url, headers={"Range": f"bytes=0-{MAX_HDR_FETCH-1}"},
                         timeout=30)
    except Exception as e:
        print(f"    header fetch failed: {e}")
        return None

    if r.status_code not in (200, 206):
        print(f"    bad status {r.status_code}")
        return None

    raw_hdr = r.content
    hdr_size, naxis1, naxis2, bitpix = parse_fits_header(raw_hdr)
    bytes_per_pix = abs(bitpix) // 8
    print(f"    FITS header: NAXIS1={naxis1} NAXIS2={naxis2} BITPIX={bitpix} hdr_size={hdr_size}", flush=True)

    rows_to_fetch = min(n_rows, naxis2)
    data_bytes = rows_to_fetch * naxis1 * bytes_per_pix
    data_padded = math.ceil(data_bytes / 2880) * 2880
    end_byte = hdr_size + data_padded - 1
    print(f"    requesting bytes 0-{end_byte} ({end_byte/1024/1024:.1f} MB)", flush=True)

    # Step 2: fetch header + image data
    try:
        r2 = requests.get(url, headers={"Range": f"bytes=0-{end_byte}"},
                          timeout=120)
    except Exception as e:
        print(f"    data fetch failed: {e}")
        return None

    if r2.status_code not in (200, 206):
        print(f"    data fetch status {r2.status_code}")
        return None

    raw = bytearray(r2.content)
    need = hdr_size + data_padded
    if len(raw) < need:
        raw += b"\x00" * (need - len(raw))

    # Patch NAXIS2 in the header to match rows_to_fetch
    patched_hdr = patch_naxis2(bytes(raw[:hdr_size]), rows_to_fetch)
    raw = bytearray(patched_hdr) + raw[hdr_size:]

    # Write to temp file so astropy can use normal file I/O
    tmp = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    tmp.write(bytes(raw))
    tmp.close()
    tmpname = tmp.name

    img = None
    try:
        with fits.open(tmpname, ignore_missing_end=True, memmap=False) as hdul:
            for ext in range(len(hdul)):
                d = hdul[ext].data
                if d is not None and d.ndim == 2:
                    img = d.astype("float32")
                    break
    except Exception as e:
        print(f"    astropy open failed: {e}")
    finally:
        os.unlink(tmpname)

    if img is None:
        return None

    return img, naxis1, naxis2


def render_and_save(img, sector, cam, ccd, png_path, naxis2_orig):
    img = img.copy()
    img[img <= 0] = np.nan
    lo, hi = np.nanpercentile(img, [1, 99.5])
    display = np.log1p(np.clip(img - lo, 0, None))

    fig, ax = plt.subplots(figsize=(6, 6 * img.shape[0] / max(img.shape[1], 1)),
                           facecolor="#020408")
    ax.imshow(display, cmap="Greys_r", origin="lower",
              interpolation="bilinear", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    label = f"S{sector} Cam{cam} CCD{ccd}"
    if img.shape[0] < naxis2_orig:
        label += f" (rows 0–{img.shape[0]})"
    ax.set_title(label, color="#c4d4ee", fontsize=9, pad=4, loc="left",
                 fontfamily="monospace")
    fig.tight_layout(pad=0)
    fig.savefig(png_path, dpi=100, bbox_inches="tight", facecolor="#020408")
    plt.close(fig)


# ── Main loop ─────────────────────────────────────────────────────────────────
total = sum(1 for s in range(S_LO, S_HI+1)
            for cam in CAM_ORDER for ccd in CCDS
            if not os.path.exists(f"{OUTDIR}/s{s:04d}_cam{cam}_ccd{ccd}.png"))
done = 0

for sec in range(S_LO, S_HI + 1):
    for cam in CAM_ORDER:
        for ccd in CCDS:
            png_path = f"{OUTDIR}/s{sec:04d}_cam{cam}_ccd{ccd}.png"
            if os.path.exists(png_path):
                continue

            done += 1
            print(f"[{done}/{total}] S{sec:02d} cam{cam} ccd{ccd} … finding URL", flush=True)
            url = find_ffi_url(sec, cam, ccd)
            if not url:
                print(f"  no URL — skip", flush=True)
                continue

            result = download_partial(url, n_rows=PARTIAL_ROWS)
            if result is None:
                print(f"  download failed — skip", flush=True)
                continue

            img, naxis1, naxis2 = result
            mb = img.nbytes / 1024 / 1024
            print(f"  got {img.shape} ({mb:.1f} MB in memory) — rendering", flush=True)
            render_and_save(img, sec, cam, ccd, png_path, naxis2)
            print(f"  → {png_path}", flush=True)

print("Done.")
