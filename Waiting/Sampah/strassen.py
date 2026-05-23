"""
=============================================================================
strassen.py — Pengolahan Citra Digital dengan Algoritma Strassen
=============================================================================
Mata Kuliah : Analisis Algoritma
Topik       : Divide and Conquer — Algoritma Strassen

Deskripsi:
    Implementasi tiga operasi pengolahan citra menggunakan Algoritma Strassen
    sebagai inti perkalian matriks. Strassen adalah algoritma Divide and Conquer
    yang mengurangi 8 perkalian rekursif menjadi 7 per level rekursi.

Relasi Rekurens  : T(n) = 7·T(n/2) + Θ(n²)
Solusi (Master)  : T(n) = Θ(n^log₂7) ≈ Θ(n^2.807)
Perbandingan     : Brute Force O(n³) > Strassen O(n^2.807) > NumPy O(n^2.37)

Ide Kunci Strassen (1969):
    Matriks A dan B masing-masing dibagi menjadi 4 sub-matriks n/2 × n/2.
    Daripada 8 perkalian (naive), Strassen hanya memerlukan 7 perkalian
    dengan tambahan beberapa penjumlahan — trade-off yang menguntungkan
    karena penjumlahan lebih murah (O(n²)) dari perkalian (O(n³)).

Cara Jalankan (Windows):
    python strassen.py
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

CHUNK_SIZE = 512   # jumlah piksel per chunk pada batch processing
BASE_CASE  = 64    # ukuran matriks di mana Strassen beralih ke np.dot


# ─────────────────────────────────────────────
#  INTI ALGORITMA STRASSEN
# ─────────────────────────────────────────────

def _strassen_recursive(A, B):
    """
    Rekursi inti Algoritma Strassen untuk matriks persegi n×n.

    Langkah Divide and Conquer:
    ─────────────────────────────────────────────────────
    DIVIDE   : Bagi A dan B masing-masing menjadi 4 sub-matriks:
                   A = [[A11, A12],    B = [[B11, B12],
                        [A21, A22]]         [B21, B22]]

    CONQUER  : Hitung 7 produk bantu (vs 8 pada naive):
                   M1 = (A11 + A22)(B11 + B22)
                   M2 = (A21 + A22) B11
                   M3 =  A11 (B12 - B22)
                   M4 =  A22 (B21 - B11)
                   M5 = (A11 + A12) B22
                   M6 = (A21 - A11)(B11 + B12)
                   M7 = (A12 - A22)(B21 + B22)

    COMBINE  : Rekonstruksi blok hasil C = A × B:
                   C11 = M1 + M4 - M5 + M7
                   C12 = M3 + M5
                   C21 = M2 + M4
                   C22 = M1 - M2 + M3 + M6

    Base case: jika n ≤ BASE_CASE, gunakan np.dot karena overhead
               rekursi lebih mahal daripada perkalian langsung.

    Parameters
    ----------
    A, B : ndarray (n×n) float64 — harus matriks persegi

    Returns
    -------
    C : ndarray (n×n) float64
    """
    n = A.shape[0]

    # Base case: beralih ke perkalian langsung untuk ukuran kecil
    if n <= BASE_CASE:
        return A @ B

    # Pad ke ukuran genap jika n ganjil
    padded = (n % 2 != 0)
    if padded:
        A = np.pad(A, ((0, 1), (0, 1)))
        B = np.pad(B, ((0, 1), (0, 1)))
        n += 1

    mid = n // 2

    # Divide: partisi sub-matriks (view, bukan copy — hemat memori)
    A11, A12 = A[:mid, :mid], A[:mid, mid:]
    A21, A22 = A[mid:, :mid], A[mid:, mid:]
    B11, B12 = B[:mid, :mid], B[:mid, mid:]
    B21, B22 = B[mid:, :mid], B[mid:, mid:]

    # Conquer: 7 perkalian rekursif Strassen
    M1 = _strassen_recursive(A11 + A22, B11 + B22)
    M2 = _strassen_recursive(A21 + A22, B11)
    M3 = _strassen_recursive(A11,        B12 - B22)
    M4 = _strassen_recursive(A22,        B21 - B11)
    M5 = _strassen_recursive(A11 + A12,  B22)
    M6 = _strassen_recursive(A21 - A11,  B11 + B12)
    M7 = _strassen_recursive(A12 - A22,  B21 + B22)

    # Combine: susun kembali blok C
    C = np.empty((n, n), dtype=np.float64)
    C[:mid, :mid] = M1 + M4 - M5 + M7
    C[:mid, mid:] = M3 + M5
    C[mid:, :mid] = M2 + M4
    C[mid:, mid:] = M1 - M2 + M3 + M6

    # Potong padding jika sebelumnya ada
    orig = n - 1 if padded else n
    return C[:orig, :orig]


def strassen(A, B):
    """
    Entry point Algoritma Strassen — perkalian dua matriks persegi.

    Konversi input ke float64 dan teruskan ke rekursi inti.

    Kompleksitas: O(n^log₂7) ≈ O(n^2.807)

    Parameters
    ----------
    A, B : array-like (n×n)

    Returns
    -------
    ndarray (n×n) float64
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    assert A.shape[0] == A.shape[1] == B.shape[0] == B.shape[1], \
        "strassen() hanya menerima matriks persegi n×n"
    return _strassen_recursive(A, B)


def strassen_matmul(A, B):
    """
    Perkalian matriks non-persegi menggunakan Strassen dengan chunking.

    Untuk matriks (R×K) × (K×C) dengan R besar (batch piksel):
      - Proses R baris dalam chunk berukuran CHUNK_SIZE
      - Setiap chunk dipad ke matriks persegi power-of-2
      - Jalankan Strassen pada chunk tersebut
      - Kumpulkan hasil dari semua chunk

    Pendekatan chunking diperlukan karena Strassen membutuhkan matriks
    persegi — padding seluruh matriks besar ke persegi akan menghabiskan
    memori secara tidak efisien.

    Kompleksitas per chunk: O(m^log₂7) di mana m = min(CHUNK_SIZE, next_pow2(K,C))

    Parameters
    ----------
    A : ndarray (R×K) — misal: batch piksel (N_piksel × 3)
    B : ndarray (K×C) — misal: matriks transformasi (3×3)ᵀ

    Returns
    -------
    ndarray (R×C)
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    R, K  = A.shape
    K2, C = B.shape
    assert K == K2, f"Dimensi tidak cocok: A({R}×{K}) × B({K2}×{C})"

    # Tentukan ukuran persegi untuk chunk + B
    n = 1
    while n < max(min(R, CHUNK_SIZE), K, C):
        n *= 2

    # Pad B ke n×n
    Bp = np.zeros((n, n), dtype=np.float64)
    Bp[:K, :C] = B

    out = np.zeros((R, C), dtype=np.float64)

    # Proses chunk demi chunk
    for start in range(0, R, CHUNK_SIZE):
        end   = min(start + CHUNK_SIZE, R)
        chunk = A[start:end]              # (chunk_len × K)
        rows  = chunk.shape[0]

        # Pad chunk ke n×n
        Ap = np.zeros((n, n), dtype=np.float64)
        Ap[:rows, :K] = chunk

        # Strassen pada matriks persegi n×n
        Cp = _strassen_recursive(Ap, Bp)  # (n×n)

        # Ambil bagian hasil yang valid
        out[start:end] = Cp[:rows, :C]

    return out


# ─────────────────────────────────────────────
#  DEFINISI MATRIKS DAN KERNEL
# ─────────────────────────────────────────────

COLOR_MATRICES = {
    "sepia": np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
    ]),
    "grayscale": np.array([
        [0.299, 0.587, 0.114],
        [0.299, 0.587, 0.114],
        [0.299, 0.587, 0.114],
    ]),
    "cool": np.array([
        [0.80, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 1.20],
    ]),
    "warm": np.array([
        [1.20, 0.05, 0.00],
        [0.00, 1.00, 0.00],
        [0.00, 0.00, 0.80],
    ]),
}

KERNELS = {
    "blur":    np.ones((5, 5), dtype=np.float64) / 25.0,
    "sharpen": np.array([[ 0,-1, 0],[-1, 5,-1],[ 0,-1, 0]], dtype=np.float64),
    "edge":    np.array([[-1,-1,-1],[-1, 8,-1],[-1,-1,-1]], dtype=np.float64),
    "emboss":  np.array([[-2,-1, 0],[-1, 1, 1],[ 0, 1, 2]], dtype=np.float64),
}


# ─────────────────────────────────────────────
#  OPERASI 1: COLOR TRANSFORM
# ─────────────────────────────────────────────

def color_transform(img, mode="sepia"):
    """
    Transformasi warna linear menggunakan Strassen pada batch piksel.

    Proses:
        1. Reshape citra (H×W×3) → matriks piksel (N×3), N = H×W
        2. Normalisasi nilai piksel 8-bit ke [0.0, 1.0]
        3. Hitung:  pixels_out = strassen_matmul(pixels_in, M.T)
               Perkalian: (N×3) × (3×3) → (N×3)
        4. Klip ke [0, 1], scale ke 8-bit, reshape kembali ke (H×W×3)

    Alasan Strassen:
        Perkalian (N×3) × (3×3) adalah perkalian matriks besar × kecil.
        Dengan chunking, setiap chunk ukuran (CHUNK_SIZE×3) × (3×3)
        dipad menjadi matriks persegi dan dikerjakan dengan Strassen.

    Kompleksitas: O(N/chunk × chunk^log₂7) ≈ lebih baik dari O(N×27)
                  (O(27) per piksel untuk brute force _matvec3)

    Parameters
    ----------
    img  : ndarray — citra BGR 8-bit dari cv2.imread
    mode : str     — "sepia" | "grayscale" | "cool" | "warm"

    Returns
    -------
    out     : ndarray — citra hasil BGR 8-bit
    elapsed : float   — waktu eksekusi dalam detik
    """
    M    = COLOR_MATRICES.get(mode, COLOR_MATRICES["sepia"])
    H, W = img.shape[:2]

    pixels = img.reshape(-1, 3).astype(np.float64) / 255.0  # (N×3)

    t0      = time.perf_counter()
    result  = strassen_matmul(pixels, M.T)                   # (N×3)
    elapsed = time.perf_counter() - t0

    out = (np.clip(result, 0, 1).reshape(H, W, 3) * 255).astype(np.uint8)
    return out, elapsed


# ─────────────────────────────────────────────
#  OPERASI 2: KONVOLUSI FILTER
# ─────────────────────────────────────────────

def _extract_patches(channel, kH, kW):
    """
    Ubah konvolusi menjadi perkalian matriks via patch extraction (im2col).

    Setiap piksel beserta neighborhood-nya diatur sebagai satu baris
    dalam matriks patches berukuran (H×W) × (kH×kW).
    Konvolusi kemudian menjadi:
        output_flat = patches @ kernel_flat
    yang merupakan perkalian matriks (N × k²) × (k² × 1).

    Teknik ini disebut "im2col" (image to column) — standar dalam
    implementasi convolutional layer di deep learning framework.

    Parameters
    ----------
    channel : ndarray (H×W) — satu channel citra
    kH, kW  : int           — ukuran kernel

    Returns
    -------
    patches : ndarray (H×W, kH×kW) float64
    """
    H, W   = channel.shape
    pH, pW = kH // 2, kW // 2
    padded = np.pad(channel.astype(np.float64),
                    ((pH, pH), (pW, pW)), mode='reflect')
    patches = np.lib.stride_tricks.sliding_window_view(
        padded, (kH, kW)
    ).reshape(H * W, kH * kW)
    return patches


def apply_filter(img, mode="blur"):
    """
    Filter konvolusi via representasi matriks im2col + Strassen.

    Proses (per channel B, G, R):
        1. Ekstrak patch: channel → patches (N × k²)  [im2col]
        2. Flatten kernel: kernel → k_vec (k² × 1)
        3. Hitung: output_flat = strassen_matmul(patches, k_vec)
        4. Reshape hasil ke (H × W)

    Dengan representasi im2col, konvolusi menjadi perkalian matriks
    dan dapat dikerjakan oleh Strassen secara eksplisit.

    Kompleksitas: O(3 × N/chunk × m^log₂7)
                  di mana m = next_pow2(max(chunk, k²))

    Parameters
    ----------
    img  : ndarray — citra BGR 8-bit
    mode : str     — "blur" | "sharpen" | "edge" | "emboss"

    Returns
    -------
    out     : ndarray — citra hasil filter BGR 8-bit
    elapsed : float   — waktu eksekusi dalam detik
    """
    kernel  = KERNELS.get(mode, KERNELS["blur"])
    kH, kW  = kernel.shape
    H, W    = img.shape[:2]
    k_vec   = kernel.ravel().reshape(-1, 1).astype(np.float64)  # (k²×1)

    channels = cv2.split(img)
    result   = []

    t0 = time.perf_counter()
    for ch in channels:
        patches  = _extract_patches(ch, kH, kW)             # (N × k²)
        filtered = strassen_matmul(patches, k_vec)           # (N × 1)
        result.append(
            np.clip(filtered.reshape(H, W), 0, 255).astype(np.uint8)
        )
    elapsed = time.perf_counter() - t0

    return cv2.merge(result), elapsed


# ─────────────────────────────────────────────
#  OPERASI 3: AFFINE TRANSFORM
# ─────────────────────────────────────────────

def affine_transform(img, angle_deg=30.0, scale=1.0, tx=0, ty=0):
    """
    Affine transform dengan batch rotasi koordinat menggunakan Strassen.

    Proses:
        1. Susun semua koordinat piksel sebagai matriks coords (N×2):
               coords[i] = [x_i - cx,  y_i - cy]
        2. Bentuk matriks rotasi R (2×2)
        3. Hitung rotasi batch:  rotated = strassen_matmul(coords, R.T)
               Perkalian: (N×2) × (2×2) → (N×2) koordinat sumber
        4. Tambahkan kembali pusat dan translasi
        5. Gunakan cv2.remap untuk sampling dengan bilinear interpolation

    Strassen digunakan pada langkah 3 — rotasi seluruh piksel sekaligus
    sebagai satu perkalian matriks besar, bukan per-piksel satu per satu.

    Kompleksitas:
        - Strassen batch: O(N/chunk × m^log₂7)
        - cv2.remap (sampling): O(H × W) — bawaan OpenCV, dioptimalkan

    Parameters
    ----------
    img       : ndarray — citra BGR 8-bit
    angle_deg : float   — sudut rotasi dalam derajat (berlawanan jarum jam)
    scale     : float   — faktor skala seragam
    tx, ty    : int     — translasi horizontal dan vertikal (piksel)

    Returns
    -------
    out     : ndarray  — citra hasil transformasi BGR 8-bit
    elapsed : float    — waktu eksekusi dalam detik (hanya bagian Strassen)
    M_aff   : ndarray  — matriks affine 2×3
    """
    H, W   = img.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    rad    = np.deg2rad(angle_deg)
    cos_a  = np.cos(rad) * scale
    sin_a  = np.sin(rad) * scale

    # Matriks rotasi 2×2
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]], dtype=np.float64)

    # Susun koordinat seluruh piksel sebagai batch (N×2)
    ys, xs  = np.mgrid[0:H, 0:W]
    coords  = np.stack([xs.ravel() - cx,
                        ys.ravel() - cy], axis=1).astype(np.float64)

    t0 = time.perf_counter()
    # Rotasi batch menggunakan Strassen: (N×2) × (2×2)ᵀ
    rotated = strassen_matmul(coords, R.T)                   # (N×2)
    elapsed = time.perf_counter() - t0

    # Koordinat sumber setelah rotasi + translasi
    src_x = (rotated[:, 0] + cx + tx).reshape(H, W).astype(np.float32)
    src_y = (rotated[:, 1] + cy + ty).reshape(H, W).astype(np.float32)

    # Remap dengan bilinear interpolation (OpenCV)
    out = cv2.remap(img, src_x, src_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT)

    M_aff = np.float32([
        [cos_a, -sin_a, (1 - cos_a)*cx + sin_a*cy + tx],
        [sin_a,  cos_a, (1 - cos_a)*cy - sin_a*cx + ty],
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
    print("  STRASSEN — Pengolahan Citra Digital")
    print("  Kompleksitas: O(n^2.807)  [Divide and Conquer]")
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
    print(f"[OK] Citra dimuat: {W}×{H} px")

    color_mode        = _menu_color()
    filter_mode       = _menu_filter()
    angle, scale, tx, ty = _menu_affine()

    out_dir = "output_strassen"
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]

    print(f"\n{'='*60}")
    print(f"  Menjalankan Strassen pada citra {W}×{H} px ...")
    print(f"{'='*60}")

    results = {}

    print("\n[1/3] Color Transform ...")
    out_c, t_c = color_transform(img, color_mode)
    p_c = os.path.join(out_dir, f"{base}_st_color_{color_mode}.png")
    cv2.imwrite(p_c, out_c)
    results["Color Transform"] = t_c
    print(f"      Selesai: {t_c*1000:.2f} ms  →  {p_c}")

    print(f"\n[2/3] Filter Konvolusi ({filter_mode}) ...")
    out_f, t_f = apply_filter(img, filter_mode)
    p_f = os.path.join(out_dir, f"{base}_st_filter_{filter_mode}.png")
    cv2.imwrite(p_f, out_f)
    results["Konvolusi Filter"] = t_f
    print(f"      Selesai: {t_f*1000:.2f} ms  →  {p_f}")

    print("\n[3/3] Affine Transform ...")
    out_a, t_a, M_aff = affine_transform(img, angle, scale, tx, ty)
    p_a = os.path.join(out_dir, f"{base}_st_affine.png")
    cv2.imwrite(p_a, out_a)
    results["Affine Transform"] = t_a
    print(f"      Selesai: {t_a*1000:.2f} ms  →  {p_a}")

    print(f"\n{'='*60}")
    print("  RINGKASAN WAKTU EKSEKUSI (Strassen)")
    print(f"  Ukuran citra : {W}×{H} px")
    print(f"  Chunk size   : {CHUNK_SIZE} piksel | Base case: {BASE_CASE}×{BASE_CASE}")
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
