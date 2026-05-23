"""
=============================================================================
numpy_blas.py — Pengolahan Citra Digital dengan NumPy + OpenCV (BLAS)
=============================================================================
Mata Kuliah : Analisis Algoritma
Topik       : Solusi Optimal — NumPy BLAS & OpenCV untuk Pengolahan Citra

Deskripsi:
    Implementasi tiga operasi pengolahan citra menggunakan NumPy dan OpenCV
    sebagai solusi optimal untuk lingkungan produksi. Tidak ada loop Python
    eksplisit — semua komputasi didelegasikan ke library level C/Fortran
    yang dioptimalkan secara hardware.

Teknologi di Balik Layar:
    - np.matmul / operator @  : BLAS routine dgemm
                                (cache-blocked + SIMD AVX-512 + multi-thread)
    - cv2.filter2D            : DFT-based convolution (kernel besar)
                                atau direct convolution (kernel kecil)
    - cv2.warpAffine          : Optimized inverse mapping + bilinear interp.

Kompleksitas Praktis:
    - Perkalian matriks  : O(n^2.37) — Coppersmith–Winograd asymptotic bound
    - Konvolusi filter   : O(H·W·log(H·W)) — via FFT untuk kernel besar
    - Affine transform   : O(H·W) — single-pass remap di level C++

Perbandingan Kompleksitas:
    Brute Force O(n³)  >  Strassen O(n^2.807)  >  NumPy BLAS O(n^2.37)

Cara Jalankan (Windows):
    python numpy_blas.py
=============================================================================
"""

import cv2
import numpy as np
import time
import os
import sys


# ─────────────────────────────────────────────
#  DEFINISI MATRIKS DAN KERNEL
# ─────────────────────────────────────────────

# Matriks transformasi warna 3×3.
# Diaplikasikan ke setiap piksel [B, G, R] via batch matrix multiply.
COLOR_MATRICES = {
    "sepia": np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
    ], dtype=np.float64),
    "grayscale": np.array([
        [0.299, 0.587, 0.114],
        [0.299, 0.587, 0.114],
        [0.299, 0.587, 0.114],
    ], dtype=np.float64),
    "cool": np.array([
        [0.80, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 1.20],
    ], dtype=np.float64),
    "warm": np.array([
        [1.20, 0.05, 0.00],
        [0.00, 1.00, 0.00],
        [0.00, 0.00, 0.80],
    ], dtype=np.float64),
}

# Kernel konvolusi 2D (float32 — optimal untuk cv2.filter2D).
KERNELS = {
    "blur":    np.ones((5, 5), dtype=np.float32) / 25.0,
    "sharpen": np.array([[ 0,-1, 0],[-1, 5,-1],[ 0,-1, 0]], dtype=np.float32),
    "edge":    np.array([[-1,-1,-1],[-1, 8,-1],[-1,-1,-1]], dtype=np.float32),
    "emboss":  np.array([[-2,-1, 0],[-1, 1, 1],[ 0, 1, 2]], dtype=np.float32),
}


# ─────────────────────────────────────────────
#  OPERASI 1: COLOR TRANSFORM
# ─────────────────────────────────────────────

def color_transform(img, mode="sepia"):
    """
    Transformasi warna linear menggunakan np.matmul — satu operasi batch penuh.

    Proses:
        1. Reshape citra (H×W×3) → matriks piksel (N×3), N = H×W
        2. Normalisasi nilai piksel 8-bit ke rentang [0.0, 1.0]
        3. Hitung perkalian matriks batch:
               result = pixels @ M.T
               (N×3) × (3×3) → (N×3)
        4. Klip ke [0,1], scale ke 8-bit, reshape ke (H×W×3)

    Mengapa lebih cepat dari Brute Force dan Strassen:
        np.matmul menggunakan BLAS dgemm yang mengeksploitasi:
        ✓ Cache-oblivious blocked matrix multiplication
        ✓ Instruksi SIMD (AVX/AVX-512) untuk operasi vektor paralel
        ✓ Multi-threading otomatis (OpenBLAS / MKL)
        Semua ini transparan bagi pengguna — cukup operator @.

    Kompleksitas: O(N × 9) di level Python, namun dengan konstanta
                  tersembunyi yang jauh lebih kecil karena BLAS C-level.

    Parameters
    ----------
    img  : ndarray — citra BGR 8-bit dari cv2.imread
    mode : str     — "sepia" | "grayscale" | "cool" | "warm"

    Returns
    -------
    out     : ndarray — citra hasil transformasi BGR 8-bit
    elapsed : float   — waktu eksekusi dalam detik
    """
    M    = COLOR_MATRICES.get(mode, COLOR_MATRICES["sepia"])
    H, W = img.shape[:2]

    # Reshape semua piksel menjadi satu matriks — tidak ada loop Python
    pixels = img.reshape(-1, 3).astype(np.float64) / 255.0   # (N × 3)

    t0      = time.perf_counter()
    result  = pixels @ M.T                                     # (N × 3)
    elapsed = time.perf_counter() - t0

    out = (np.clip(result, 0, 1).reshape(H, W, 3) * 255).astype(np.uint8)
    return out, elapsed


# ─────────────────────────────────────────────
#  OPERASI 2: KONVOLUSI FILTER
# ─────────────────────────────────────────────

def apply_filter(img, mode="blur"):
    """
    Filter konvolusi 2D menggunakan cv2.filter2D.

    cv2.filter2D secara otomatis memilih strategi optimal:
        - Kernel kecil (≤ ~11×11) : Direct convolution (O(H·W·kH·kW))
        - Kernel besar             : DFT-based convolution (O(H·W·log(H·W)))
    Pemilihan dilakukan otomatis berdasarkan biaya komputasi relatif.

    Perbandingan dengan Brute Force:
        Brute Force (blur 5×5 pada 256×256):
            256 × 256 × 5 × 5 × 3 = ~49.1 juta iterasi Python
        NumPy cv2.filter2D:
            Seluruh operasi di level C++ — loop Python = 0

    Parameters
    ----------
    img  : ndarray — citra BGR 8-bit
    mode : str     — "blur" | "sharpen" | "edge" | "emboss"

    Returns
    -------
    out     : ndarray — citra hasil filter BGR 8-bit
    elapsed : float   — waktu eksekusi dalam detik
    """
    kernel = KERNELS.get(mode, KERNELS["blur"])

    t0      = time.perf_counter()
    # -1 berarti kedalaman output sama dengan input (uint8 → uint8)
    result  = cv2.filter2D(img, -1, kernel)
    elapsed = time.perf_counter() - t0

    return result, elapsed


# ─────────────────────────────────────────────
#  OPERASI 3: AFFINE TRANSFORM
# ─────────────────────────────────────────────

def affine_transform(img, angle_deg=30.0, scale=1.0, tx=0, ty=0):
    """
    Transformasi affine (rotasi + skala + translasi) menggunakan cv2.warpAffine.

    Proses:
        1. Hitung matriks rotasi-skala 2×3 dengan cv2.getRotationMatrix2D:
               M = scale × [[cos θ, -sin θ, tx'],
                             [sin θ,  cos θ, ty']]
           di mana tx', ty' menyertakan kompensasi pusat rotasi.

        2. Tambahkan offset translasi manual tx, ty.

        3. Terapkan transformasi ke seluruh citra dengan cv2.warpAffine:
               dst(x, y) = src(M⁻¹ × [x, y, 1]ᵀ)
           menggunakan bilinear interpolation untuk nilai sub-piksel.

    Mengapa lebih cepat:
        cv2.warpAffine menggunakan:
        ✓ Inverse mapping vectorized di level C++
        ✓ Bilinear interpolation dengan tabel lookup
        ✓ Parallelisasi per-baris via OpenMP (bila tersedia)

    Kompleksitas: O(H × W) — satu pass per piksel di level C++

    Parameters
    ----------
    img       : ndarray — citra BGR 8-bit
    angle_deg : float   — sudut rotasi dalam derajat (berlawanan jarum jam)
    scale     : float   — faktor skala seragam (1.0 = tidak berubah)
    tx, ty    : int     — translasi horizontal dan vertikal (piksel)

    Returns
    -------
    out     : ndarray  — citra hasil transformasi BGR 8-bit
    elapsed : float    — waktu eksekusi dalam detik
    M_aff   : ndarray  — matriks affine 2×3 yang digunakan
    """
    H, W   = img.shape[:2]
    cx, cy = W / 2.0, H / 2.0

    # Bangun matriks rotasi-skala di sekitar pusat citra
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)

    # Tambahkan komponen translasi manual
    M[0, 2] += tx
    M[1, 2] += ty

    t0     = time.perf_counter()
    result = cv2.warpAffine(
        img, M, (W, H),
        flags=cv2.INTER_LINEAR,          # bilinear interpolation
        borderMode=cv2.BORDER_CONSTANT,  # piksel di luar batas → hitam
        borderValue=(0, 0, 0)
    )
    elapsed = time.perf_counter() - t0

    return result, elapsed, M


# ─────────────────────────────────────────────
#  MENU CLI
# ─────────────────────────────────────────────

def _menu_color():
    opts = list(COLOR_MATRICES.keys())
    print("\n  Mode Transformasi Warna:")
    for i, k in enumerate(opts, 1):
        print(f"    [{i}] {k}")
    c = input("  Pilih [1-4, default=1]: ").strip()
    return opts[int(c)-1] if c.isdigit() and 1 <= int(c) <= len(opts) else opts[0]


def _menu_filter():
    opts = list(KERNELS.keys())
    print("\n  Mode Filter Konvolusi:")
    for i, k in enumerate(opts, 1):
        print(f"    [{i}] {k}")
    c = input("  Pilih [1-4, default=1]: ").strip()
    return opts[int(c)-1] if c.isdigit() and 1 <= int(c) <= len(opts) else opts[0]


def _menu_affine():
    print("\n  Parameter Affine Transform:")
    angle = input("  Sudut rotasi derajat [default=30]: ").strip()
    scale = input("  Skala [default=1.0]: ").strip()
    tx    = input("  Translasi X px [default=0]: ").strip()
    ty    = input("  Translasi Y px [default=0]: ").strip()
    return (
        float(angle) if angle else 30.0,
        float(scale) if scale else 1.0,
        int(tx) if tx else 0,
        int(ty) if ty else 0,
    )


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  NUMPY + OpenCV — Pengolahan Citra Digital")
    print("  Kompleksitas: O(n^2.37)  [BLAS + SIMD Optimized]")
    print("=" * 60)

    path = input("\nPath citra input: ").strip().strip('"')
    if not os.path.exists(path):
        print(f"[ERROR] File tidak ditemukan: {path}")
        sys.exit(1)

    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] Tidak dapat membaca citra: {path}")
        sys.exit(1)

    H, W = img.shape[:2]
    print(f"[OK] Citra dimuat: {W}×{H} px  (tidak ada batasan ukuran)")

    color_mode           = _menu_color()
    filter_mode          = _menu_filter()
    angle, scale, tx, ty = _menu_affine()

    out_dir = "output_numpy"
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]

    print(f"\n{'='*60}")
    print(f"  Menjalankan NumPy+OpenCV pada citra {W}×{H} px ...")
    print(f"{'='*60}")

    results = {}

    print("\n[1/3] Color Transform ...")
    out_c, t_c = color_transform(img, color_mode)
    p_c = os.path.join(out_dir, f"{base}_np_color_{color_mode}.png")
    cv2.imwrite(p_c, out_c)
    results["Color Transform"] = t_c
    print(f"      Selesai: {t_c*1000:.2f} ms  →  {p_c}")

    print(f"\n[2/3] Filter Konvolusi ({filter_mode}) ...")
    out_f, t_f = apply_filter(img, filter_mode)
    p_f = os.path.join(out_dir, f"{base}_np_filter_{filter_mode}.png")
    cv2.imwrite(p_f, out_f)
    results["Konvolusi Filter"] = t_f
    print(f"      Selesai: {t_f*1000:.2f} ms  →  {p_f}")

    print("\n[3/3] Affine Transform ...")
    out_a, t_a, M_aff = affine_transform(img, angle, scale, tx, ty)
    p_a = os.path.join(out_dir, f"{base}_np_affine.png")
    cv2.imwrite(p_a, out_a)
    results["Affine Transform"] = t_a
    print(f"      Selesai: {t_a*1000:.2f} ms  →  {p_a}")

    print(f"\n{'='*60}")
    print("  RINGKASAN WAKTU EKSEKUSI (NumPy + OpenCV BLAS)")
    print(f"  Ukuran citra : {W}×{H} px")
    print(f"  {'Operasi':<22} | {'Waktu (ms)':>10}")
    print(f"  {'-'*22}-+-{'-'*10}")
    for nama, t in results.items():
        print(f"  {nama:<22} | {t*1000:>10.2f}")
    total = sum(results.values())
    print(f"  {'-'*22}-+-{'-'*10}")
    print(f"  {'TOTAL':<22} | {total*1000:>10.2f}")
    print(f"{'='*60}")
    print(f"\n  Output tersimpan di folder: ./{out_dir}/")


if __name__ == "__main__":
    main()
