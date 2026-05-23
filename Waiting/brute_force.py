"""
=============================================================================
brute_force.py — Pengolahan Citra Digital dengan Pendekatan Brute Force
=============================================================================
Mata Kuliah : Analisis Algoritma
Topik       : Strategi Brute Force dalam Pengolahan Citra Digital

Deskripsi:
    Implementasi tiga operasi pengolahan citra menggunakan pendekatan
    Brute Force murni — semua komputasi dilakukan dengan loop Python
    eksplisit tanpa optimasi library apapun.

Kompleksitas Waktu:
    - Color Transform : O(H × W × 9)        — 9 operasi per piksel
    - Konvolusi       : O(H × W × kH × kW)  — sliding window penuh
    - Affine Transform: O(H × W × 6)        — inverse mapping per piksel

Keterangan:
    Karena kompleksitas tinggi, citra di-resize otomatis ke MAX_DIM px
    agar waktu eksekusi brute force masih dapat diamati dengan wajar.
    Ukuran asli tetap digunakan oleh strassen.py dan numpy_blas.py.

Cara Jalankan (Windows):
    python brute_force.py
=============================================================================
"""

import cv2
import numpy as np
import time
import os
import sys


# ─────────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────────

MAX_DIM = 128   # batas sisi terpanjang untuk brute force


# ─────────────────────────────────────────────
#  DEFINISI MATRIKS DAN KERNEL
# ─────────────────────────────────────────────

# Matriks transformasi warna 3×3
# Setiap piksel [B, G, R] dikalikan dengan matriks ini.
COLOR_MATRICES = {
    "sepia": [
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
    ],
    "grayscale": [
        [0.299, 0.587, 0.114],
        [0.299, 0.587, 0.114],
        [0.299, 0.587, 0.114],
    ],
    "cool": [
        [0.80, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 1.20],
    ],
    "warm": [
        [1.20, 0.05, 0.00],
        [0.00, 1.00, 0.00],
        [0.00, 0.00, 0.80],
    ],
}

# Kernel konvolusi 2D
# Setiap elemen kernel menentukan bobot piksel tetangga.
KERNELS = {
    "blur": [[1/25]*5 for _ in range(5)],                          # rata-rata 5×5
    "sharpen": [[0,-1,0], [-1,5,-1], [0,-1,0]],                   # perkuat tepi
    "edge": [[-1,-1,-1], [-1,8,-1], [-1,-1,-1]],                  # deteksi tepi
    "emboss": [[-2,-1,0], [-1,1,1], [0,1,2]],                     # efek emboss
}


# ─────────────────────────────────────────────
#  HELPER: PERKALIAN MATRIKS BRUTE FORCE
# ─────────────────────────────────────────────

def _matvec3(M, v):
    """
    Kalikan matriks 3×3 dengan vektor 3×1 — satu operasi piksel.

    Ini adalah inti dari color transform: setiap piksel [B, G, R]
    dikalikan secara eksplisit dengan 9 operasi perkalian dan 6 penjumlahan.

    Kompleksitas: O(9) = O(1) per piksel
    """
    return [
        M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
        M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
        M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2],
    ]


# ─────────────────────────────────────────────
#  OPERASI 1: COLOR TRANSFORM
# ─────────────────────────────────────────────

def color_transform(img, mode="sepia"):
    """
    Transformasi warna linear dengan perkalian matriks per piksel (Brute Force).

    Proses:
        Untuk setiap piksel pada posisi (y, x):
            pixel_out = M_color × pixel_in

        Di mana pixel_in = [B, G, R] dinormalisasi ke [0, 1],
        dan M_color adalah matriks 3×3 yang mendefinisikan transformasi.

    Implementasi menggunakan dua loop bersarang (y, x) dan fungsi _matvec3
    yang melakukan 9 perkalian secara eksplisit — tidak ada vectorization.

    Kompleksitas Waktu : O(H × W × 9) ≈ O(H × W)
    Kompleksitas Ruang : O(H × W × 3) untuk array output

    Parameters
    ----------
    img  : ndarray  — citra BGR 8-bit hasil cv2.imread
    mode : str      — "sepia" | "grayscale" | "cool" | "warm"

    Returns
    -------
    out     : ndarray — citra hasil transformasi BGR 8-bit
    elapsed : float   — waktu eksekusi dalam detik
    """
    M    = COLOR_MATRICES.get(mode, COLOR_MATRICES["sepia"])
    H, W = img.shape[:2]
    out  = np.zeros_like(img)

    t0 = time.perf_counter()

    for y in range(H):
        for x in range(W):
            # Normalisasi piksel 8-bit ke rentang [0.0, 1.0]
            px  = [img[y, x, c] / 255.0 for c in range(3)]
            res = _matvec3(M, px)
            # Klip ke [0, 255] dan kembalikan ke 8-bit
            for c in range(3):
                out[y, x, c] = max(0, min(255, int(res[c] * 255)))

    elapsed = time.perf_counter() - t0
    return out, elapsed


# ─────────────────────────────────────────────
#  OPERASI 2: KONVOLUSI FILTER
# ─────────────────────────────────────────────

def apply_filter(img, mode="blur"):
    """
    Filter konvolusi 2D dengan sliding window Brute Force.

    Proses (untuk setiap channel warna):
        Untuk setiap piksel (y, x):
            output[y, x] = Σ kernel[ky, kx] × channel[y+ky-pH, x+kx-pW]
                           (ky=0..kH-1, kx=0..kW-1)

        Di mana pH = kH//2 dan pW = kW//2 adalah padding pusat kernel.
        Piksel di luar batas citra diabaikan (zero-padding implisit).

    Implementasi menggunakan empat loop bersarang eksplisit (y, x, ky, kx)
    — ini adalah bentuk konvolusi paling mendasar tanpa optimasi apapun.

    Kompleksitas Waktu : O(H × W × kH × kW × C)
                         Contoh: 128×128 × 5×5 × 3 ≈ 12.3 juta operasi
    Kompleksitas Ruang : O(H × W × C)

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
    kH     = len(kernel)
    kW     = len(kernel[0])
    pH, pW = kH // 2, kW // 2
    H, W   = img.shape[:2]
    out    = np.zeros_like(img, dtype=np.float64)

    t0 = time.perf_counter()

    # Loop per channel (B, G, R)
    for c in range(3):
        for y in range(H):
            for x in range(W):
                acc = 0.0
                # Sliding window — iterasi setiap elemen kernel
                for ky in range(kH):
                    for kx in range(kW):
                        ny = y + ky - pH
                        nx = x + kx - pW
                        # Cek batas citra (zero-padding di luar batas)
                        if 0 <= ny < H and 0 <= nx < W:
                            acc += img[ny, nx, c] * kernel[ky][kx]
                out[y, x, c] = acc

    elapsed = time.perf_counter() - t0
    return np.clip(out, 0, 255).astype(np.uint8), elapsed


# ─────────────────────────────────────────────
#  OPERASI 3: AFFINE TRANSFORM
# ─────────────────────────────────────────────

def affine_transform(img, angle_deg=30.0, scale=1.0, tx=0, ty=0):
    """
    Transformasi affine (rotasi + skala + translasi) dengan inverse mapping
    per piksel secara Brute Force.

    Proses:
        1. Bentuk matriks rotasi-skala 2×2:
               R = scale × [[cos θ, -sin θ],
                             [sin θ,  cos θ]]

        2. Untuk setiap piksel tujuan (x_dst, y_dst):
               [x_src]   = R⁻¹ × ([x_dst - cx] - [tx]) + [cx]
               [y_src]            ([y_dst - cy]   [ty])   [cy]

        3. Ambil nilai piksel dari koordinat sumber dengan nearest-neighbor.

    Inverse mapping digunakan agar tidak ada lubang (hole) pada output.
    Semua komputasi dilakukan dengan loop eksplisit dan aritmetika float manual.

    Kompleksitas Waktu : O(H × W × 6) ≈ O(H × W)
    Kompleksitas Ruang : O(H × W × C)

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
    rad    = angle_deg * 3.141592653589793 / 180.0
    cos_a  = (2.718281828459045 ** (0+1j*rad)).real * scale   # manual cos
    sin_a  = (2.718281828459045 ** (0+1j*rad)).imag * scale   # manual sin
    out    = np.zeros_like(img)

    # Invers matriks rotasi 2×2: R⁻¹ = (1/scale²) × [[cos, sin], [-sin, cos]]
    det     = cos_a * cos_a + sin_a * sin_a   # = scale²
    inv_cos =  cos_a / det
    inv_sin = -sin_a / det

    t0 = time.perf_counter()

    for y_dst in range(H):
        for x_dst in range(W):
            # Koordinat relatif terhadap pusat, setelah dikurangi translasi
            dx = (x_dst - tx) - cx
            dy = (y_dst - ty) - cy

            # Terapkan inverse rotation
            x_src = inv_cos * dx - inv_sin * dy + cx
            y_src = inv_sin * dx + inv_cos * dy + cy

            # Nearest-neighbor sampling
            ix = int(round(x_src))
            iy = int(round(y_src))
            if 0 <= ix < W and 0 <= iy < H:
                out[y_dst, x_dst] = img[iy, ix]

    elapsed = time.perf_counter() - t0

    # Matriks affine 2×3 untuk referensi / perbandingan
    M_aff = np.float32([
        [cos_a, -sin_a, (1-cos_a)*cx + sin_a*cy + tx],
        [sin_a,  cos_a, (1-cos_a)*cy - sin_a*cx + ty],
    ])
    return out, elapsed, M_aff


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
    print("  BRUTE FORCE — Pengolahan Citra Digital")
    print("  Kompleksitas: O(n³) per operasi matriks")
    print("=" * 60)

    # Input citra
    path = input("\nPath citra input: ").strip().strip('"')
    if not os.path.exists(path):
        print(f"[ERROR] File tidak ditemukan: {path}")
        sys.exit(1)

    img_full = cv2.imread(path)
    if img_full is None:
        print(f"[ERROR] Tidak dapat membaca citra: {path}")
        sys.exit(1)

    # Resize untuk brute force
    H, W = img_full.shape[:2]
    if max(H, W) > MAX_DIM:
        ratio = MAX_DIM / max(H, W)
        img   = cv2.resize(img_full, (int(W*ratio), int(H*ratio)))
        print(f"\n[INFO] Citra di-resize: {W}×{H} → {img.shape[1]}×{img.shape[0]} px")
        print(f"       (Brute Force dibatasi {MAX_DIM}px agar waktu dapat diamati)")
    else:
        img = img_full.copy()

    # Menu operasi
    color_mode = _menu_color()
    filter_mode = _menu_filter()
    angle, scale, tx, ty = _menu_affine()

    # Output dir
    out_dir = "output_brute_force"
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]

    # Jalankan ketiga operasi
    print(f"\n{'='*60}")
    print(f"  Menjalankan Brute Force pada citra {img.shape[1]}×{img.shape[0]} px ...")
    print(f"{'='*60}")

    results = {}

    # 1. Color Transform
    print("\n[1/3] Color Transform ...")
    out_c, t_c = color_transform(img, color_mode)
    p_c = os.path.join(out_dir, f"{base}_bf_color_{color_mode}.png")
    cv2.imwrite(p_c, out_c)
    results["Color Transform"] = t_c
    print(f"      Selesai: {t_c*1000:.2f} ms  →  {p_c}")

    # 2. Konvolusi
    print(f"\n[2/3] Filter Konvolusi ({filter_mode}) ...")
    out_f, t_f = apply_filter(img, filter_mode)
    p_f = os.path.join(out_dir, f"{base}_bf_filter_{filter_mode}.png")
    cv2.imwrite(p_f, out_f)
    results["Konvolusi Filter"] = t_f
    print(f"      Selesai: {t_f*1000:.2f} ms  →  {p_f}")

    # 3. Affine
    print("\n[3/3] Affine Transform ...")
    out_a, t_a, M_aff = affine_transform(img, angle, scale, tx, ty)
    p_a = os.path.join(out_dir, f"{base}_bf_affine.png")
    cv2.imwrite(p_a, out_a)
    results["Affine Transform"] = t_a
    print(f"      Selesai: {t_a*1000:.2f} ms  →  {p_a}")

    # Ringkasan
    print(f"\n{'='*60}")
    print("  RINGKASAN WAKTU EKSEKUSI (Brute Force)")
    print(f"  Ukuran citra : {img.shape[1]}×{img.shape[0]} px")
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
