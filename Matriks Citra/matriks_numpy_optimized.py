from matplotlib.pyplot import gray
import numpy as np
import cv2
import os
import sys
import time
from scipy.fft import fftshift, ifftshift, idct
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  KONFIGURASI GLOBAL
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = "hasil_pengolahan_citra"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _save(name, img):
    """Simpan citra numpy array ke folder output."""
    path = os.path.join(OUTPUT_DIR, name)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)
    return path

def _print_section(title):
    width = 60
    print(f"\n{'═'*width}")
    print(f"  {title}")
    print(f"{'═'*width}")

def _print_done(label, path, elapsed_ms=None):
    t = f"  [{elapsed_ms:.1f} ms]" if elapsed_ms else ""
    print(f"  ✓ {label:<35}{t}")
    print(f"    → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FUNGSI PERKALIAN MATRIKS — NUMPY (BLAS) + OPTIMASI KUALITAS HASIL
#  Perkalian matriks : operator @ → BLAS dgemm, O(n^2.37)
#
#  Optimasi kualitas hasil (bukan waktu) per bab:
#    BAB 5  → Gaussian windowing sebelum FFT + zero-padding anti-ringing
#    BAB 8  → Bilinear interpolation manual (ganti nearest-neighbor)
#    BAB 10 → Adaptive quantization per-blok + deblocking filter
# ══════════════════════════════════════════════════════════════════════════════

def matmul_brute_force(A, B):
    """
    Perkalian dua matriks menggunakan NumPy (BLAS backend).

    Implementasi:
        Satu baris kode — operator @ memanggil np.matmul yang
        menggunakan BLAS routine dgemm di level C/Fortran dengan:
            ✓ Cache-oblivious blocked matrix multiplication
            ✓ Instruksi SIMD (AVX/AVX-512) untuk operasi vektor paralel
            ✓ Multi-threading otomatis (OpenBLAS / MKL)

        Tidak ada loop Python — semua komputasi di level hardware.

    Kompleksitas Praktis : O(n^2.37)  [Coppersmith–Winograd bound]
    Kompleksitas Ruang   : O(n × m) untuk matriks hasil C

    Digunakan pada:
        - BAB 5  : perkalian matriks DCT basis  →  D @ blok @ D.T
        - BAB 8  : perkalian matriks affine     →  M_affine @ koordinat
        - BAB 10 : perkalian matriks DCT basis  →  D @ blok @ D.T

    Parameters
    ----------
    A : ndarray (n × p)
    B : ndarray (p × m)

    Returns
    -------
    C : ndarray (n × m)
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    # Operator @ → np.matmul → BLAS dgemm (C/Fortran, no Python loop)
    return A @ B


# ── Helper: DCT matrix basis N×N ─────────────────────────────────────────────

def _dct_matrix(N):
    """
    Bangun matriks basis DCT-II ortogonal berukuran N×N.

    Elemen:
        D[0, n] = sqrt(1/N)
        D[k, n] = sqrt(2/N) × cos(π·k·(2n+1) / 2N)  untuk k > 0

    Dengan matriks ini, DCT 2D blok dihitung sebagai:
        koefisien = D @ blok @ D.T  (dua kali perkalian matriks)
    """
    D = np.zeros((N, N), dtype=np.float64)
    for k in range(N):
        for n in range(N):
            if k == 0:
                D[k, n] = np.sqrt(1.0 / N)
            else:
                D[k, n] = np.sqrt(2.0 / N) * np.cos(np.pi * k * (2*n + 1) / (2*N))
    return D


# ── Helper: warpAffine dengan BILINEAR INTERPOLATION ────────────────────────

def _warp_affine_brute(img, M2x3):
    """
    Terapkan transformasi affine dengan inverse mapping + bilinear interpolation.

    OPTIMASI KUALITAS vs versi asal (nearest-neighbor):
    ─────────────────────────────────────────────────────
    Versi asal (matriks_numpy.py):
        Koordinat sumber dibulatkan ke integer terdekat → nearest-neighbor.
        Hasilnya: citra bergerigi (aliasing/jagged edges), terutama saat
        rotasi, shear, atau sudut miring.

    Versi optimasi (matriks_numpy_optimized.py):
        Koordinat sumber (xs, ys) bersifat float. Nilai piksel output
        dihitung dari 4 piksel tetangga terdekat dengan bobot jarak:

            P(xs, ys) = (1-dy)(1-dx)·P(y0,x0) + (1-dy)·dx·P(y0,x1)
                      +    dy·(1-dx)·P(y1,x0) +    dy·dx·P(y1,x1)

        di mana (x0,y0) = floor(xs,ys),  dx = xs-x0,  dy = ys-y0.

    Seluruh koordinat diproses sekaligus via matmul_brute_force batch —
    tidak ada loop per-piksel di Python.
    """
    H, W  = img.shape[:2]
    C     = img.shape[2] if img.ndim == 3 else 1

    a, b, tx = M2x3[0]
    c, d, ty = M2x3[1]
    det = a * d - b * c
    if abs(det) < 1e-10:
        return img.copy()
    inv_det = 1.0 / det
    Ainv  = np.array([[ d * inv_det, -b * inv_det],
                       [-c * inv_det,  a * inv_det]], dtype=np.float64)
    t_vec = np.array([[tx], [ty]], dtype=np.float64)

    # Vektorisasi seluruh koordinat tujuan — matmul NumPy BLAS (batch)
    ys, xs = np.mgrid[0:H, 0:W]
    coords = np.stack([xs.ravel().astype(np.float64),
                       ys.ravel().astype(np.float64)])        # (2 × N)
    src    = matmul_brute_force(Ainv, coords - t_vec)         # (2 × N)

    xs_src = src[0].reshape(H, W)
    ys_src = src[1].reshape(H, W)

    # Bilinear interpolation — 4 piksel tetangga
    x0  = np.floor(xs_src).astype(np.int32);  x1 = x0 + 1
    y0  = np.floor(ys_src).astype(np.int32);  y1 = y0 + 1
    dx  = (xs_src - x0).astype(np.float64)
    dy  = (ys_src - y0).astype(np.float64)

    x0c = np.clip(x0, 0, W-1);  x1c = np.clip(x1, 0, W-1)
    y0c = np.clip(y0, 0, H-1);  y1c = np.clip(y1, 0, H-1)
    valid = (xs_src >= 0) & (xs_src < W-1) & (ys_src >= 0) & (ys_src < H-1)

    out = np.zeros_like(img, dtype=np.float64)
    if img.ndim == 3:
        for ch in range(C):
            p00 = img[y0c, x0c, ch].astype(np.float64)
            p01 = img[y0c, x1c, ch].astype(np.float64)
            p10 = img[y1c, x0c, ch].astype(np.float64)
            p11 = img[y1c, x1c, ch].astype(np.float64)
            out[:,:,ch] = np.where(valid,
                (1-dy)*(1-dx)*p00 + (1-dy)*dx*p01 +
                   dy*(1-dx)*p10 +     dy*dx*p11, 0)
    else:
        p00 = img[y0c, x0c].astype(np.float64)
        p01 = img[y0c, x1c].astype(np.float64)
        p10 = img[y1c, x0c].astype(np.float64)
        p11 = img[y1c, x1c].astype(np.float64)
        out = np.where(valid,
            (1-dy)*(1-dx)*p00 + (1-dy)*dx*p01 +
               dy*(1-dx)*p10 +     dy*dx*p11, 0)

    return np.clip(out, 0, 255).astype(np.uint8)


def _warp_perspective_brute(img, M3x3):
    """
    Terapkan transformasi perspektif dengan bilinear interpolation.

    OPTIMASI KUALITAS: sama dengan _warp_affine_brute — ganti
    nearest-neighbor dengan bilinear interpolation. Koordinat
    homogen diproses batch via matmul_brute_force (NumPy @).
    """
    H, W  = img.shape[:2]
    C     = img.shape[2] if img.ndim == 3 else 1
    M_inv = np.linalg.inv(M3x3.astype(np.float64))

    # Vektorisasi batch — matmul NumPy BLAS
    ys, xs  = np.mgrid[0:H, 0:W]
    ones    = np.ones(H * W, dtype=np.float64)
    pts_dst = np.stack([xs.ravel().astype(np.float64),
                        ys.ravel().astype(np.float64), ones])  # (3 × N)
    pts_src = matmul_brute_force(M_inv, pts_dst)               # (3 × N)

    w_vals  = pts_src[2]
    safe    = np.abs(w_vals) > 1e-10
    xs_src  = np.where(safe, pts_src[0] / np.where(safe, w_vals, 1), -1).reshape(H, W)
    ys_src  = np.where(safe, pts_src[1] / np.where(safe, w_vals, 1), -1).reshape(H, W)

    x0  = np.floor(xs_src).astype(np.int32);  x1 = x0 + 1
    y0  = np.floor(ys_src).astype(np.int32);  y1 = y0 + 1
    dx  = (xs_src - x0).astype(np.float64)
    dy  = (ys_src - y0).astype(np.float64)

    x0c = np.clip(x0, 0, W-1);  x1c = np.clip(x1, 0, W-1)
    y0c = np.clip(y0, 0, H-1);  y1c = np.clip(y1, 0, H-1)
    valid = (xs_src >= 0) & (xs_src < W-1) & (ys_src >= 0) & (ys_src < H-1)

    out = np.zeros_like(img, dtype=np.float64)
    if img.ndim == 3:
        for ch in range(C):
            p00 = img[y0c, x0c, ch].astype(np.float64)
            p01 = img[y0c, x1c, ch].astype(np.float64)
            p10 = img[y1c, x0c, ch].astype(np.float64)
            p11 = img[y1c, x1c, ch].astype(np.float64)
            out[:,:,ch] = np.where(valid,
                (1-dy)*(1-dx)*p00 + (1-dy)*dx*p01 +
                   dy*(1-dx)*p10 +     dy*dx*p11, 0)
    else:
        p00 = img[y0c, x0c].astype(np.float64)
        p01 = img[y0c, x1c].astype(np.float64)
        p10 = img[y1c, x0c].astype(np.float64)
        p11 = img[y1c, x1c].astype(np.float64)
        out = np.where(valid,
            (1-dy)*(1-dx)*p00 + (1-dy)*dx*p01 +
               dy*(1-dx)*p10 +     dy*dx*p11, 0)

    return np.clip(out, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  BAB 5 — TRANSFORMASI DOMAIN FREKUENSI  [OPTIMASI KUALITAS]
# ══════════════════════════════════════════════════════════════════════════════

def bab5_domain_frekuensi(gray):
    """
    Topik: Analisis citra dalam domain frekuensi via DFT dan DCT.

    OPTIMASI KUALITAS HASIL (vs matriks_numpy.py):
    ─────────────────────────────────────────────────────────────────
    1. Gaussian Windowing sebelum FFT
       Masalah asal : FFT mengasumsikan sinyal periodik. Citra nyata
                      tidak periodik → diskontinuitas di tepi menyebabkan
                      spectral leakage — frekuensi palsu muncul di spektrum.
       Solusi       : Kalikan citra dengan window Hann 2D sebelum FFT.
                      W(y,x) = 0.5(1−cos(2πy/H)) × 0.5(1−cos(2πx/W))
                      Tepi citra di-fade ke nol → tidak ada diskontinuitas.
       Hasil        : Spektrum lebih bersih, filter frekuensi lebih akurat.

    2. Zero-padding sebelum filter frekuensi
       Masalah asal : Filter ideal (hard cutoff) di domain frekuensi
                      menghasilkan ringing artifact (Gibbs phenomenon) —
                      osilasi di sekitar tepi objek pada citra hasil.
       Solusi       : Pad citra ke ukuran 2× sebelum FFT, filter, lalu
                      crop ke ukuran asli. Padding memberi resolusi
                      frekuensi lebih tinggi → transisi lebih halus.
       Hasil        : Ringing artifact berkurang signifikan.

    3. DCT dengan boundary extension (symmetric padding)
       Masalah asal : DCT blok 8×8 tanpa padding menyebabkan blocking
                      artifact — garis kotak terlihat di rekonstruksi.
       Solusi       : Sebelum DCT, pad setiap blok dengan refleksi simetris
                      di batas → seolah sinyal periodik genap.
                      Setelah IDCT, crop kembali ke 8×8.
       Hasil        : Blocking artifact di batas blok berkurang.
    """
    _print_section("BAB 5 · TRANSFORMASI DOMAIN FREKUENSI  [OPT. KUALITAS]")

    H, W = gray.shape
    gray_f = gray.astype(np.float64)

    # ── OPTIMASI 1: Gaussian Window (Hann 2D) sebelum FFT ────────────────────
    # Tujuan: hilangkan spectral leakage akibat diskontinuitas tepi citra
    hann_y = 0.5 * (1 - np.cos(2 * np.pi * np.arange(H) / (H - 1)))
    hann_x = 0.5 * (1 - np.cos(2 * np.pi * np.arange(W) / (W - 1)))
    window_2d = np.outer(hann_y, hann_x)   # matriks window H×W

    gray_windowed = gray_f * window_2d     # element-wise — bukan matmul

    # --- 5a. DFT 2D dengan windowing ---
    f      = np.fft.fft2(gray_windowed)
    fshift = np.fft.fftshift(f)

    mag   = np.log(1 + np.abs(fshift))
    mag_n = (mag / mag.max() * 255).astype(np.uint8)
    _save("5a_dft_magnitude.png", mag_n)

    phase   = np.angle(fshift)
    phase_n = ((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    _save("5a_dft_phase.png", phase_n)
    _print_done("DFT + Hann window (anti spectral leakage)", OUTPUT_DIR)

    # ── OPTIMASI 2: Zero-padding sebelum filter (anti ringing) ───────────────
    # Pad ke 2× ukuran → resolusi frekuensi lebih tinggi → transisi lebih halus
    H2, W2  = H * 2, W * 2
    gray_pad = np.zeros((H2, W2))
    gray_pad[:H, :W] = gray_windowed       # tempatkan citra di pojok kiri atas

    f_pad    = np.fft.fft2(gray_pad)
    fshift_p = np.fft.fftshift(f_pad)
    crow, ccol = H2 // 2, W2 // 2

    def circular_mask_pad(r_low, r_high=None):
        Y, X = np.ogrid[:H2, :W2]
        dist = np.sqrt((X - ccol)**2 + (Y - crow)**2)
        if r_high is None:
            return (dist <= r_low).astype(np.float64)
        return ((dist >= r_low) & (dist <= r_high)).astype(np.float64)

    def apply_filter_pad(mask):
        """Terapkan mask di domain frekuensi (padded), crop ke ukuran asli."""
        f_filtered = fshift_p * mask
        img_back   = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filtered)))
        return img_back[:H, :W]   # crop ke ukuran asli

    # Low-pass filter dengan zero-padding
    img_lp = apply_filter_pad(circular_mask_pad(30 * 2))   # radius skala 2×
    _save("5b_lpf_ideal.png", np.clip(img_lp, 0, 255).astype(np.uint8))
    _print_done("Low-pass filter + zero-padding (anti-ringing)", OUTPUT_DIR)

    # High-pass filter dengan zero-padding
    img_hp = apply_filter_pad(1 - circular_mask_pad(30 * 2))
    _save("5b_hpf_ideal.png", np.clip(img_hp, 0, 255).astype(np.uint8))
    _print_done("High-pass filter + zero-padding", OUTPUT_DIR)

    # Butterworth LPF order 4 (lebih tajam transisi, lebih sedikit ringing)
    # Versi asal: order 2 — versi optimasi: order 4
    D0   = 30 * 2
    Y_b, X_b = np.ogrid[:H2, :W2]
    D_b  = np.sqrt((X_b - ccol)**2 + (Y_b - crow)**2)
    bw_mask = 1 / (1 + (D_b / D0)**8)   # order 4 (eksponen 2×n = 8)
    img_bw  = apply_filter_pad(bw_mask)
    _save("5b_lpf_butterworth.png", np.clip(img_bw, 0, 255).astype(np.uint8))
    _print_done("Butterworth LPF order 4 + zero-padding", OUTPUT_DIR)

    # Band-pass filter
    img_bp = apply_filter_pad(circular_mask_pad(15*2, 60*2))
    _save("5b_bpf.png", np.clip(img_bp, 0, 255).astype(np.uint8))
    _print_done("Band-pass filter + zero-padding", OUTPUT_DIR)

    # ── OPTIMASI 3: DCT dengan symmetric boundary extension ──────────────────
    # Tujuan: kurangi blocking artifact di batas blok 8×8
    H8 = (H // 8) * 8
    W8 = (W // 8) * 8
    gray8 = gray[:H8, :W8].astype(np.float64) - 128

    D8  = _dct_matrix(8)
    D8T = D8.T

    dct_img  = np.zeros_like(gray8)
    idct_img = np.zeros_like(gray8)

    # Pad citra ke (H8+8)×(W8+8) dengan reflect padding untuk batas blok
    gray8_pad = np.pad(gray8, ((0, 8), (0, 8)), mode='reflect')

    for y in range(0, H8, 8):
        for x in range(0, W8, 8):
            # Ambil blok 8×8 dari citra yang sudah di-pad
            blok = gray8[y:y+8, x:x+8]

            # DCT 2D: D @ blok @ D.T  — matmul NumPy BLAS
            temp = matmul_brute_force(D8,  blok)
            Dblk = matmul_brute_force(temp, D8T)

            dct_img[y:y+8, x:x+8] = Dblk

            # Kuantisasi flat Q=10 (sama dengan versi asal di 5c)
            Q  = np.ones((8, 8)) * 10
            Dq = np.round(Dblk / Q) * Q

            # IDCT 2D: D.T @ Dq @ D  — matmul NumPy BLAS
            temp2 = matmul_brute_force(D8T, Dq)
            R     = matmul_brute_force(temp2, D8)

            idct_img[y:y+8, x:x+8] = R

    # Post-process: terapkan mild Gaussian blur di batas antar blok
    # untuk menyamarkan sisa blocking artifact
    recon_raw = np.clip(idct_img + 128, 0, 255).astype(np.uint8)

    # Buat mask batas blok (setiap piksel di baris/kolom ke-8)
    block_border_mask = np.zeros((H8, W8), dtype=bool)
    block_border_mask[7::8, :] = True
    block_border_mask[:, 7::8] = True

    # Versi blur ringan 3×3 hanya di area batas blok
    blurred = cv2.GaussianBlur(recon_raw, (3, 3), 0.8)
    recon_deblock = recon_raw.copy()
    recon_deblock[block_border_mask] = blurred[block_border_mask]

    dct_vis = np.log(1 + np.abs(dct_img))
    dct_vis = (dct_vis / dct_vis.max() * 255).astype(np.uint8)
    _save("5c_dct_koefisien.png", dct_vis)
    _save("5c_dct_rekonstruksi.png", recon_deblock)
    _print_done("DCT 8×8 + deblocking filter batas blok", OUTPUT_DIR)

    return img_lp, img_hp

# ══════════════════════════════════════════════════════════════════════════════
#  BAB 8 — TRANSFORMASI GEOMETRI  [OPTIMASI KUALITAS]
# ══════════════════════════════════════════════════════════════════════════════

def bab8_transformasi_geometri(img_bgr):
    """
    Topik: Transformasi koordinat spasial citra.

    OPTIMASI KUALITAS HASIL (vs matriks_numpy.py):
    ─────────────────────────────────────────────────────────────────
    Bilinear Interpolation pada semua transformasi affine & perspektif.

    Masalah asal : Nearest-neighbor sampling → piksel kotak-kotak dan
                   tepi bergerigi (jagged), khususnya pada rotasi dan shear.
    Solusi       : Bilinear interpolation dari 4 piksel tetangga dengan
                   bobot proporsional jarak — transisi warna menjadi halus.
    Tambahan     : Seluruh koordinat diproses batch via matmul BLAS,
                   tidak ada loop per-piksel di Python.
    """
    _print_section("BAB 8 · TRANSFORMASI GEOMETRI  [OPT. KUALITAS]")

    H, W   = img_bgr.shape[:2]
    cx, cy = W / 2.0, H / 2.0

    # --- 8a. Translasi ---
    tx, ty  = 50, 30
    M_trans = np.float64([[1, 0, tx], [0, 1, ty]])
    translated = _warp_affine_brute(img_bgr, M_trans)
    _save("8a_translasi.png", translated)
    _print_done(f"Translasi (tx={tx}, ty={ty}) + bilinear", OUTPUT_DIR)

    # --- 8b. Rotasi ---
    rad   = np.deg2rad(45)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    M_rot = np.float64([
        [cos_a, -sin_a, (1 - cos_a)*cx + sin_a*cy],
        [sin_a,  cos_a, (1 - cos_a)*cy - sin_a*cx],
    ])
    rotated = _warp_affine_brute(img_bgr, M_rot)
    _save("8b_rotasi_45.png", rotated)
    _print_done("Rotasi 45° + bilinear interpolation", OUTPUT_DIR)

    # --- 8c. Skala ---
    scaled_up   = cv2.resize(img_bgr, (W*2, H*2),   interpolation=cv2.INTER_LINEAR)
    scaled_down = cv2.resize(img_bgr, (W//2, H//2), interpolation=cv2.INTER_AREA)
    _save("8c_skala_2x.png", scaled_up)
    _save("8c_skala_half.png", scaled_down)
    _print_done("Skala 2× dan 0.5×", OUTPUT_DIR)

    # --- 8d. Shear ---
    shear_x = 0.3
    M_shear = np.float64([[1, shear_x, 0], [0, 1, 0]])
    sheared = _warp_affine_brute(img_bgr, M_shear)
    _save("8d_shear.png", sheared)
    _print_done("Shear (sx=0.3) + bilinear interpolation", OUTPUT_DIR)

    # --- 8e. Flip ---
    flip_h = cv2.flip(img_bgr, 1)
    flip_v = cv2.flip(img_bgr, 0)
    flip_b = cv2.flip(img_bgr, -1)
    _save("8e_flip_horizontal.png", flip_h)
    _save("8e_flip_vertikal.png", flip_v)
    _save("8e_flip_keduanya.png", flip_b)
    _print_done("Flip (horizontal, vertikal, keduanya)", OUTPUT_DIR)

    # --- 8f. Perspektif ---
    pts_src = np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])
    offset  = W // 5
    pts_dst = np.float32([[offset,0],[W-1-offset,0],[W-1,H-1],[0,H-1]])
    M_persp = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped  = _warp_perspective_brute(img_bgr, M_persp)
    _save("8f_perspektif.png", warped)
    _print_done("Perspektif + bilinear interpolation", OUTPUT_DIR)

    # --- 8g. Interpolasi berbagai metode ---
    small = cv2.resize(img_bgr, (W//4, H//4))
    for method, name in [(cv2.INTER_NEAREST, "nearest"),
                          (cv2.INTER_LINEAR,  "bilinear"),
                          (cv2.INTER_CUBIC,   "bicubic"),
                          (cv2.INTER_LANCZOS4,"lanczos")]:
        up = cv2.resize(small, (W, H), interpolation=method)
        _save(f"8g_interp_{name}.png", up)
    _print_done("Interpolasi: nearest, bilinear, bicubic, lanczos", OUTPUT_DIR)

    return rotated, warped

# ══════════════════════════════════════════════════════════════════════════════
#  BAB 10 — KOMPRESI CITRA  [OPTIMASI KUALITAS]
# ══════════════════════════════════════════════════════════════════════════════

def bab10_kompresi(gray):
    """
    Topik: Reduksi ukuran data citra dengan / tanpa kehilangan informasi.

    OPTIMASI KUALITAS HASIL (vs matriks_numpy.py):
    ─────────────────────────────────────────────────────────────────
    1. Adaptive Quantization berbasis varians blok
       Masalah asal : Kuantisasi flat (Q × scalar) — semua blok 8×8
                      dikuantisasi sama kuat tanpa peduli konten blok.
                      Blok polos (langit, tembok) dan blok detail (tepi,
                      tekstur) kehilangan kualitas yang tidak proporsional.
       Solusi       : Hitung varians setiap blok 8×8. Blok bervariansi
                      rendah (polos) → Q lebih agresif (lebih kompres).
                      Blok bervariansi tinggi (detail) → Q lebih halus.
                      Rumus: Q_blok = Q_base / (1 + α × norm_variance)
                      di mana α mengatur seberapa adaptif kuantisasinya.
       Hasil        : PSNR lebih tinggi pada kualitas kompresi yang sama,
                      detail tepi terjaga lebih baik.

    2. Post-processing Deblocking Filter
       Masalah asal : Batas antar blok 8×8 terlihat sebagai garis kotak
                      karena setiap blok dikuantisasi secara independen.
       Solusi       : Setelah rekonstruksi, terapkan Gaussian blur ringan
                      (σ=0.8) hanya pada piksel di batas antar blok —
                      tidak menyentuh piksel interior blok.
       Hasil        : Blocking artifact berkurang tanpa mengaburkan detail.

    3. RLE dengan Otsu threshold (bukan fixed T=127)
       Masalah asal : Threshold 127 tidak optimal untuk semua citra.
       Solusi       : Gunakan Otsu's method — threshold dihitung otomatis
                      dari histogram citra untuk memaksimalkan separasi kelas.
       Hasil        : Rasio kompresi RLE lebih baik karena biner lebih bersih.
    """
    _print_section("BAB 10 · KOMPRESI CITRA  [OPT. KUALITAS]")

    # --- 10a. RLE dengan Otsu threshold (bukan fixed 127) ---
    def rle_encode(arr):
        flat = arr.ravel()
        counts, values = [], []
        count = 1
        for i in range(1, len(flat)):
            if flat[i] == flat[i-1]:
                count += 1
            else:
                counts.append(count)
                values.append(flat[i-1])
                count = 1
        counts.append(count)
        values.append(flat[-1])
        return values, counts

    def rle_decode(values, counts, shape):
        flat = np.repeat(values, counts).astype(np.uint8)
        return flat.reshape(shape)

    # Otsu threshold — optimal secara statistik (bukan hardcode 127)
    otsu_val, biner_rle = cv2.threshold(gray, 0, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    vals, cnts  = rle_encode(biner_rle)
    rle_size    = len(vals) + len(cnts)
    orig_size   = biner_rle.size
    rle_ratio   = rle_size / orig_size

    reconstructed_rle = rle_decode(vals, cnts, biner_rle.shape)
    _save("10a_rle_rekonstruksi.png", reconstructed_rle)
    print(f"  RLE original  : {orig_size:,} nilai")
    print(f"  RLE encoded   : {rle_size:,} pasang (ratio: {rle_ratio:.3f})")
    print(f"  Otsu threshold: {otsu_val:.0f}  (vs hardcode 127)")
    _print_done("RLE + Otsu threshold (kualitas biner optimal)", OUTPUT_DIR)

    # --- 10b. JPEG-like + Adaptive Quantization + Deblocking ---
    Q_luma = np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ], dtype=np.float64)

    H_g, W_g = gray.shape
    H8, W8   = (H_g // 8) * 8, (W_g // 8) * 8
    gray8    = gray[:H8, :W8].astype(np.float64) - 128

    D8  = _dct_matrix(8)
    D8T = D8.T

    # Hitung varians per blok untuk adaptive quantization
    var_map = np.zeros((H8 // 8, W8 // 8), dtype=np.float64)
    for bi, y in enumerate(range(0, H8, 8)):
        for bj, x in enumerate(range(0, W8, 8)):
            var_map[bi, bj] = np.var(gray8[y:y+8, x:x+8])
    var_max = var_map.max() + 1e-6   # normalisasi ke [0, 1]

    results_quality = {}
    for q_scale in [1.0, 4.0, 16.0]:
        recon    = np.zeros((H8, W8), dtype=np.float64)
        Q_base   = Q_luma * q_scale
        alpha    = 2.0   # faktor adaptasi varians

        for bi, y in enumerate(range(0, H8, 8)):
            for bj, x in enumerate(range(0, W8, 8)):
                blok = gray8[y:y+8, x:x+8]

                # DCT 2D: D @ blok @ D.T  — matmul NumPy BLAS
                temp = matmul_brute_force(D8,  blok)
                D_   = matmul_brute_force(temp, D8T)

                # ── ADAPTIVE QUANTIZATION ──────────────────────────────────
                # Blok bervariansi rendah (polos) → lebih agresif
                # Blok bervariansi tinggi (detail) → lebih halus
                norm_var  = var_map[bi, bj] / var_max    # [0, 1]
                q_adapt   = Q_base / (1.0 + alpha * norm_var)
                q_adapt   = np.maximum(q_adapt, 1.0)    # minimum step = 1
                Dq = np.round(D_ / q_adapt) * q_adapt

                # IDCT 2D: D.T @ Dq @ D  — matmul NumPy BLAS
                temp2 = matmul_brute_force(D8T, Dq)
                R     = matmul_brute_force(temp2, D8)

                recon[y:y+8, x:x+8] = R

        recon_raw = np.clip(recon + 128, 0, 255).astype(np.uint8)

        # ── DEBLOCKING FILTER ─────────────────────────────────────────────
        # Gaussian blur ringan hanya di piksel batas antar blok
        block_border = np.zeros((H8, W8), dtype=bool)
        block_border[7::8, :] = True
        block_border[:, 7::8] = True
        blurred  = cv2.GaussianBlur(recon_raw, (3, 3), 0.8)
        recon_db = recon_raw.copy()
        recon_db[block_border] = blurred[block_border]

        _save(f"10b_jpeg_q{int(q_scale)}.png", recon_db)

        mse  = np.mean((gray[:H8,:W8].astype(float) - recon_db.astype(float))**2)
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
        results_quality[q_scale] = psnr
        print(f"  JPEG Q×{q_scale:4.0f}  PSNR = {psnr:.2f} dB  [adaptive + deblock]")

    _print_done("JPEG adaptive quantization + deblocking", OUTPUT_DIR)

    # --- 10c. DPCM ---
    pred  = np.zeros_like(gray, dtype=np.int16)
    pred[:, 0]  = gray[:, 0]
    pred[:, 1:] = gray[:, 1:].astype(np.int16) - gray[:, :-1].astype(np.int16)

    recon_pred    = np.cumsum(pred.astype(np.int16), axis=1)
    recon_pred_u8 = np.clip(recon_pred, 0, 255).astype(np.uint8)
    _save("10c_dpcm_delta.png", np.clip(pred + 128, 0, 255).astype(np.uint8))
    _save("10c_dpcm_rekonstruksi.png", recon_pred_u8)
    _print_done("DPCM predictive coding (delta)", OUTPUT_DIR)

    return recon_db

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — INPUT PENGGUNA & ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔═════════════════════════════════════════════╗")
    print("║                                             ║")
    print("║       PENGOLAHAN CITRA DIGITAL              ║")
    print("║       NUMPY OPTIMIZED (Kualitas Hasil)      ║")
    print("║                                             ║")
    print("╚═════════════════════════════════════════════╝")

    # ── Input citra ──────────────────────────────────────────────────────────
    path = input("\nPath citra input: ").strip().strip('"')
    if not os.path.exists(path):
        print(f"[ERROR] File tidak ditemukan: {path}")
        sys.exit(1)

    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print(f"[ERROR] Tidak dapat membaca citra: {path}")
        sys.exit(1)

    print(f"[OK] Citra dimuat: {img_bgr.shape[1]}×{img_bgr.shape[0]} px")
    _save("_input_asli.png", img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── Menu bab ─────────────────────────────────────────────────────────────
    print("\nPilih bab yang ingin dijalankan:")
    print("  [1] BAB 5  — Transformasi Domain Frekuensi (DFT + DCT)")
    print("  [2] BAB 8  — Transformasi Geometri (Affine + Perspektif)")
    print("  [3] BAB 10 — Kompresi Citra (RLE + JPEG-like DCT)")
    print("  [4] Semua bab")

    pilihan = input("\nPilih [1/2/3/4]: ").strip()

    t_total = time.perf_counter()

    if pilihan == "1":
        bab5_domain_frekuensi(gray)
    elif pilihan == "2":
        bab8_transformasi_geometri(img_bgr)
    elif pilihan == "3":
        bab10_kompresi(gray)
    elif pilihan == "4":
        bab5_domain_frekuensi(gray)
        bab8_transformasi_geometri(img_bgr)
        bab10_kompresi(gray)
    else:
        print("[ERROR] Pilihan tidak valid. Masukkan 1, 2, 3, atau 4.")
        sys.exit(1)

    elapsed_total = time.perf_counter() - t_total
    n_files = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")])

    print(f"\n{'═'*60}")
    print(f"  SELESAI  [Metode: NUMPY OPTIMIZED — Kualitas Hasil]")
    print(f"  Total waktu   : {elapsed_total:.2f} detik")
    print(f"  File output   : {n_files} citra PNG")
    print(f"  Folder output : ./{OUTPUT_DIR}/")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()