"""
digital_image_processing.py
============================
Pengolahan Citra Digital — Implementasi Lengkap dengan NumPy
Mencakup seluruh materi standar mata kuliah Pengolahan Citra Digital.

Cara menjalankan:
    python digital_image_processing.py
"""

import numpy as np
import cv2
import os
import sys
import time
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift, ifftshift, dct, idct
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
#  BAB 1 — AKUISISI & REPRESENTASI CITRA
# ══════════════════════════════════════════════════════════════════════════════

def bab1_akuisisi_representasi(img_bgr):
    """
    Topik: Representasi digital citra — piksel, channel, bit-depth.
    Citra digital adalah matriks 2D (grayscale) atau 3D (warna) berisi
    nilai integer 0–255 (8-bit per channel).
    """
    _print_section("BAB 1 · AKUISISI & REPRESENTASI CITRA")

    H, W, C = img_bgr.shape
    print(f"  Dimensi      : {W} × {H} piksel")
    print(f"  Channel      : {C}  (B, G, R)")
    print(f"  Bit-depth    : {img_bgr.dtype}  (8-bit per channel)")
    print(f"  Total piksel : {H*W:,}")
    print(f"  Ukuran array : {img_bgr.nbytes/1024:.1f} KB")

    # --- 1a. Pisah channel BGR ---
    B, G, R = cv2.split(img_bgr)
    _save("1a_channel_B.png", B)
    _save("1a_channel_G.png", G)
    _save("1a_channel_R.png", R)
    _print_done("Pemisahan channel B, G, R", OUTPUT_DIR)

    # --- 1b. Konversi ruang warna ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    # YCbCr manual dengan NumPy (matriks konversi standar ITU-R BT.601)
    img_f  = img_bgr.astype(np.float64)
    M_ycbcr = np.array([
        [ 0.299,   0.587,   0.114 ],
        [-0.16874,-0.33126, 0.500 ],
        [ 0.500,  -0.41869,-0.08131]
    ])
    pix = img_f[:,:,::-1].reshape(-1, 3)   # BGR → RGB, lalu flatten
    ycbcr_flat = pix @ M_ycbcr.T
    ycbcr_flat[:, 1:] += 128
    ycbcr = np.clip(ycbcr_flat, 0, 255).reshape(H, W, 3).astype(np.uint8)

    _save("1b_grayscale.png", gray)
    _save("1b_hsv.png", hsv)
    _save("1b_lab.png", lab)
    _save("1b_ycbcr_numpy.png", ycbcr)
    _print_done("Konversi BGR→Gray, HSV, LAB, YCbCr", OUTPUT_DIR)

    # --- 1c. Representasi bit-plane ---
    # Setiap bit dari nilai piksel dipisahkan sebagai citra biner
    bitplanes = []
    for bit in range(8):
        plane = ((gray >> bit) & 1) * 255
        bitplanes.append(plane.astype(np.uint8))
        _save(f"1c_bitplane_{bit}.png", plane.astype(np.uint8))
    _print_done("Bit-plane slicing (bit 0–7)", OUTPUT_DIR)

    # --- 1d. Citra biner threshold manual ---
    thresh = (gray > 127).astype(np.uint8) * 255
    _save("1d_biner_threshold.png", thresh)
    _print_done("Konversi ke citra biner", OUTPUT_DIR)

    return gray, hsv, lab


# ══════════════════════════════════════════════════════════════════════════════
#  BAB 2 — TRANSFORMASI INTENSITAS & PENINGKATAN KONTRAS
# ══════════════════════════════════════════════════════════════════════════════

def bab2_transformasi_intensitas(gray):
    """
    Topik: Operasi point-wise pada nilai intensitas piksel.
    Setiap piksel diubah menggunakan fungsi transformasi s = T(r).
    """
    _print_section("BAB 2 · TRANSFORMASI INTENSITAS")

    # --- 2a. Negatif citra: s = L-1 - r ---
    negatif = 255 - gray
    _save("2a_negatif.png", negatif)
    _print_done("Negatif citra  (s = 255 - r)", OUTPUT_DIR)

    # --- 2b. Log transform: s = c × log(1 + r) ---
    c   = 255 / np.log(1 + 255)
    log_t = (c * np.log(1.0 + gray.astype(np.float64))).astype(np.uint8)
    _save("2b_log_transform.png", log_t)
    _print_done("Log transform   (s = c·log(1+r))", OUTPUT_DIR)

    # --- 2c. Power-law / Gamma correction: s = c × r^γ ---
    for gamma, label in [(0.4, "terang"), (1.0, "normal"), (2.5, "gelap")]:
        normalized = gray.astype(np.float64) / 255.0
        gamma_img  = np.power(normalized, gamma) * 255
        _save(f"2c_gamma_{label}.png", gamma_img)
    _print_done("Gamma correction (γ = 0.4, 1.0, 2.5)", OUTPUT_DIR)

    # --- 2d. Piecewise linear transform (contrast stretching) ---
    r1, s1, r2, s2 = 50, 0, 200, 255
    lut = np.zeros(256, dtype=np.float64)
    for r in range(256):
        if r < r1:
            lut[r] = (s1 / (r1 + 1e-6)) * r
        elif r < r2:
            lut[r] = ((s2 - s1) / (r2 - r1)) * (r - r1) + s1
        else:
            lut[r] = ((255 - s2) / (255 - r2 + 1e-6)) * (r - r2) + s2
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    contrast_stretch = lut[gray]
    _save("2d_contrast_stretch.png", contrast_stretch)
    _print_done("Contrast stretching (piecewise)", OUTPUT_DIR)

    # --- 2e. Thresholding berbagai metode ---
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _save("2e_otsu.png", otsu)

    # Adaptive threshold manual (local mean - C)
    C_val   = 5
    ksize   = 15
    blur    = cv2.blur(gray.astype(np.float64), (ksize, ksize))
    adaptive_np = ((gray.astype(np.float64)) > (blur - C_val)).astype(np.uint8) * 255
    _save("2e_adaptive_thresh_numpy.png", adaptive_np)
    _print_done("Thresholding Otsu & Adaptive", OUTPUT_DIR)

    # --- 2f. Brightness & Contrast adjustment ---
    alpha, beta = 1.5, 30
    bright = np.clip(alpha * gray.astype(np.float64) + beta, 0, 255).astype(np.uint8)
    _save("2f_brightness_contrast.png", bright)
    _print_done("Brightness/Contrast (α·r + β)", OUTPUT_DIR)

    return contrast_stretch


# ══════════════════════════════════════════════════════════════════════════════
#  BAB 3 — HISTOGRAM & EQUALISASI
# ══════════════════════════════════════════════════════════════════════════════

def bab3_histogram(gray):
    """
    Topik: Histogram sebagai distribusi intensitas piksel.
    Histogram equalization meratakan distribusi untuk meningkatkan kontras.
    """
    _print_section("BAB 3 · HISTOGRAM & EQUALISASI")

    H, W = gray.shape
    N    = H * W

    # --- 3a. Hitung histogram manual dengan NumPy ---
    hist = np.zeros(256, dtype=np.int64)
    flat = gray.ravel()
    for val in flat:
        hist[val] += 1
    # Setara dengan: hist, _ = np.histogram(gray, bins=256, range=(0,255))

    print(f"  Intensitas min   : {gray.min()}")
    print(f"  Intensitas max   : {gray.max()}")
    print(f"  Intensitas mean  : {gray.mean():.2f}")
    print(f"  Std deviasi      : {gray.std():.2f}")
    _print_done("Perhitungan histogram manual", OUTPUT_DIR)

    # --- 3b. Histogram equalization manual ---
    # CDF (Cumulative Distribution Function)
    cdf = np.cumsum(hist)
    cdf_min = cdf[cdf > 0][0]

    # Fungsi transformasi HE: s = round((cdf(r) - cdf_min) / (N - cdf_min) * 255)
    lut_he = np.round((cdf - cdf_min) / (N - cdf_min) * 255).astype(np.uint8)
    lut_he = np.clip(lut_he, 0, 255)
    gray_eq = lut_he[gray]
    _save("3b_histogram_equalized.png", gray_eq)
    _print_done("Histogram equalization (manual)", OUTPUT_DIR)

    # --- 3c. CLAHE (Contrast Limited Adaptive HE) via OpenCV ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    _save("3c_clahe.png", gray_clahe)
    _print_done("CLAHE (adaptive equalization)", OUTPUT_DIR)

    # --- 3d. Histogram matching (specification) ---
    # Cocokkan histogram citra ke distribusi target (Gaussian)
    target_hist = np.exp(-((np.arange(256) - 128)**2) / (2 * 40**2))
    target_hist = (target_hist / target_hist.sum() * N).astype(np.int64)
    target_cdf  = np.cumsum(target_hist)

    # Bangun LUT dengan matching CDF
    lut_match = np.zeros(256, dtype=np.uint8)
    for r in range(256):
        best = np.argmin(np.abs(target_cdf - cdf[r]))
        lut_match[r] = best
    gray_matched = lut_match[gray]
    _save("3d_histogram_matched.png", gray_matched)
    _print_done("Histogram specification/matching", OUTPUT_DIR)

    return gray_eq


# ══════════════════════════════════════════════════════════════════════════════
#  BAB 4 — FILTER SPASIAL (DOMAIN SPATIAL)
# ══════════════════════════════════════════════════════════════════════════════

def bab4_filter_spasial(gray, img_bgr):
    """
    Topik: Konvolusi 2D dengan kernel dalam domain spasial.
    Output = Σ kernel(s,t) × f(x+s, y+t) untuk semua (s,t).
    """
    _print_section("BAB 4 · FILTER SPASIAL")

    def convolve2d_numpy(img, kernel):
        """Konvolusi 2D dengan NumPy — menggunakan scipy.ndimage.convolve."""
        return ndimage.convolve(img.astype(np.float64), kernel,
                                mode='reflect')

    # --- 4a. Filter Perata (Smoothing / Low-pass) ---
    # Mean filter (box filter)
    k_mean = np.ones((5, 5), dtype=np.float64) / 25
    mean_f = np.clip(convolve2d_numpy(gray, k_mean), 0, 255).astype(np.uint8)
    _save("4a_mean_filter.png", mean_f)
    _print_done("Mean filter 5×5", OUTPUT_DIR)

    # Gaussian filter
    # Kernel Gaussian: G(x,y) = (1/2πσ²) × exp(-(x²+y²)/2σ²)
    def gaussian_kernel(size, sigma):
        ax   = np.arange(-(size//2), size//2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        k    = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return k / k.sum()

    k_gauss  = gaussian_kernel(5, 1.5)
    gauss_f  = np.clip(convolve2d_numpy(gray, k_gauss), 0, 255).astype(np.uint8)
    _save("4b_gaussian_filter.png", gauss_f)
    _print_done("Gaussian filter 5×5 σ=1.5", OUTPUT_DIR)

    # Median filter (non-linear — bukan konvolusi)
    median_f = ndimage.median_filter(gray, size=5)
    _save("4b_median_filter.png", median_f.astype(np.uint8))
    _print_done("Median filter 5×5 (non-linear)", OUTPUT_DIR)

    # --- 4b. Filter Penajam (Sharpening / High-pass) ---
    # Laplacian
    k_lap = np.array([[ 0,-1, 0],
                       [-1, 4,-1],
                       [ 0,-1, 0]], dtype=np.float64)
    lap   = convolve2d_numpy(gray, k_lap)
    sharp_lap = np.clip(gray.astype(np.float64) - lap, 0, 255).astype(np.uint8)
    _save("4c_laplacian_sharp.png", sharp_lap)
    _print_done("Laplacian sharpening", OUTPUT_DIR)

    # Unsharp masking: f_sharp = f + k × (f - f_blur)
    k_um  = 1.5
    sharp_um = np.clip(gray.astype(np.float64) + k_um * (gray.astype(np.float64) - gauss_f.astype(np.float64)), 0, 255).astype(np.uint8)
    _save("4d_unsharp_masking.png", sharp_um)
    _print_done("Unsharp masking (k=1.5)", OUTPUT_DIR)

    # High-boost filter: f_hb = A×f - f_blur
    A = 2.0
    highboost = np.clip(A * gray.astype(np.float64) - gauss_f.astype(np.float64), 0, 255).astype(np.uint8)
    _save("4d_highboost_filter.png", highboost)
    _print_done("High-boost filter (A=2.0)", OUTPUT_DIR)

    # --- 4c. Filter Gradien (Edge-sensitive) ---
    # Prewitt
    k_px = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float64)
    k_py = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=np.float64)
    gx_p = convolve2d_numpy(gray, k_px)
    gy_p = convolve2d_numpy(gray, k_py)
    prewitt = np.clip(np.sqrt(gx_p**2 + gy_p**2), 0, 255).astype(np.uint8)
    _save("4e_prewitt.png", prewitt)
    _print_done("Prewitt operator", OUTPUT_DIR)

    # Sobel
    k_sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float64)
    k_sy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float64)
    gx_s = convolve2d_numpy(gray, k_sx)
    gy_s = convolve2d_numpy(gray, k_sy)
    sobel = np.clip(np.sqrt(gx_s**2 + gy_s**2), 0, 255).astype(np.uint8)
    _save("4e_sobel.png", sobel)
    _print_done("Sobel operator", OUTPUT_DIR)

    # Roberts cross
    k_r1 = np.array([[1,0],[0,-1]], dtype=np.float64)
    k_r2 = np.array([[0,1],[-1,0]], dtype=np.float64)
    gr1  = convolve2d_numpy(gray, k_r1)
    gr2  = convolve2d_numpy(gray, k_r2)
    roberts = np.clip(np.abs(gr1) + np.abs(gr2), 0, 255).astype(np.uint8)
    _save("4e_roberts.png", roberts)
    _print_done("Roberts cross operator", OUTPUT_DIR)

    # Canny (OpenCV — deteksi tepi multi-stage)
    canny = cv2.Canny(gray, 50, 150)
    _save("4f_canny.png", canny)
    _print_done("Canny edge detector", OUTPUT_DIR)

    return gauss_f, sobel


# ══════════════════════════════════════════════════════════════════════════════
#  BAB 5 — TRANSFORMASI DOMAIN FREKUENSI
# ══════════════════════════════════════════════════════════════════════════════

def bab5_domain_frekuensi(gray):
    """
    Topik: Analisis citra dalam domain frekuensi via DFT dan DCT.
    Komponen frekuensi rendah = informasi umum (smooth).
    Komponen frekuensi tinggi = detail, tepi, noise.
    """
    _print_section("BAB 5 · TRANSFORMASI DOMAIN FREKUENSI")

    H, W = gray.shape

    # --- 5a. DFT 2D dengan NumPy (FFT) ---
    f     = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)                          # geser DC ke tengah
    mag   = np.log(1 + np.abs(fshift))
    mag_n = (mag / mag.max() * 255).astype(np.uint8)
    _save("5a_dft_magnitude.png", mag_n)

    phase = np.angle(fshift)
    phase_n = ((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    _save("5a_dft_phase.png", phase_n)
    _print_done("DFT — spektrum magnitude & fase", OUTPUT_DIR)

    # --- 5b. Filter Frekuensi (Low-pass, High-pass, Band-pass) ---
    crow, ccol = H // 2, W // 2

    def circular_mask(H, W, r_low, r_high=None):
        Y, X = np.ogrid[:H, :W]
        dist = np.sqrt((X - ccol)**2 + (Y - crow)**2)
        if r_high is None:
            return (dist <= r_low).astype(np.float64)
        return ((dist >= r_low) & (dist <= r_high)).astype(np.float64)

    # Ideal Low-pass filter
    mask_lp  = circular_mask(H, W, 30)
    f_lp     = fshift * mask_lp
    img_lp   = np.abs(np.fft.ifft2(np.fft.ifftshift(f_lp)))
    _save("5b_lpf_ideal.png", np.clip(img_lp, 0, 255).astype(np.uint8))
    _print_done("Low-pass filter  (r=30, ideal)", OUTPUT_DIR)

    # Ideal High-pass filter
    mask_hp  = 1 - mask_lp
    f_hp     = fshift * mask_hp
    img_hp   = np.abs(np.fft.ifft2(np.fft.ifftshift(f_hp)))
    _save("5b_hpf_ideal.png", np.clip(img_hp, 0, 255).astype(np.uint8))
    _print_done("High-pass filter (r=30, ideal)", OUTPUT_DIR)

    # Butterworth Low-pass filter (order n=2)
    n_bw = 2
    D0   = 30
    Y_b, X_b = np.ogrid[:H, :W]
    D    = np.sqrt((X_b - ccol)**2 + (Y_b - crow)**2)
    bw_lp_mask = 1 / (1 + (D / D0)**(2 * n_bw))
    f_bw = fshift * bw_lp_mask
    img_bw = np.abs(np.fft.ifft2(np.fft.ifftshift(f_bw)))
    _save("5b_lpf_butterworth.png", np.clip(img_bw, 0, 255).astype(np.uint8))
    _print_done("Butterworth LPF  (n=2, D0=30)", OUTPUT_DIR)

    # Band-pass filter
    mask_bp  = circular_mask(H, W, 15, 60)
    f_bp     = fshift * mask_bp
    img_bp   = np.abs(np.fft.ifft2(np.fft.ifftshift(f_bp)))
    _save("5b_bpf.png", np.clip(img_bp, 0, 255).astype(np.uint8))
    _print_done("Band-pass filter (r=15–60)", OUTPUT_DIR)

    # --- 5c. DCT 2D (Discrete Cosine Transform) — dasar JPEG ---
    # Proses blok 8×8 seperti standar JPEG
    H8 = (H // 8) * 8
    W8 = (W // 8) * 8
    gray8 = gray[:H8, :W8].astype(np.float64) - 128   # level shift

    dct_img  = np.zeros_like(gray8)
    idct_img = np.zeros_like(gray8)

    for y in range(0, H8, 8):
        for x in range(0, W8, 8):
            blok = gray8[y:y+8, x:x+8]
            D    = dct(dct(blok, axis=0, norm='ortho'), axis=1, norm='ortho')
            dct_img[y:y+8, x:x+8] = D
            # Kuantisasi sederhana (faktor Q=10) → simulasi kompresi
            Q    = np.ones((8,8)) * 10
            Dq   = np.round(D / Q) * Q
            R    = idct(idct(Dq, axis=1, norm='ortho'), axis=0, norm='ortho')
            idct_img[y:y+8, x:x+8] = R

    dct_vis = np.log(1 + np.abs(dct_img))
    dct_vis = (dct_vis / dct_vis.max() * 255).astype(np.uint8)
    _save("5c_dct_koefisien.png", dct_vis)

    recon = np.clip(idct_img + 128, 0, 255).astype(np.uint8)
    _save("5c_dct_rekonstruksi.png", recon)
    _print_done("DCT 2D blok 8×8 + rekonstruksi", OUTPUT_DIR)

    return img_lp, img_hp


# ══════════════════════════════════════════════════════════════════════════════
#  BAB 6 — MORFOLOGI CITRA
# ══════════════════════════════════════════════════════════════════════════════

def bab6_morfologi(gray):
    """
    Topik: Operasi morfologi berbasis teori himpunan.
    Structuring Element (SE) menentukan bentuk operasi.
    Berlaku pada citra biner maupun grayscale.
    """
    _print_section("BAB 6 · MORFOLOGI CITRA")

    # Buat citra biner dari Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _save("6_input_biner.png", binary)

    # Structuring Element
    se_rect  = cv2.getStructuringElement(cv2.MORPH_RECT,    (5, 5))
    se_ellip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    se_cross = cv2.getStructuringElement(cv2.MORPH_CROSS,   (5, 5))

    # --- 6a. Operasi dasar morfologi biner ---
    # Erosion: menyusutkan objek, menghilangkan noise kecil
    eroded   = cv2.erode(binary, se_rect, iterations=1)
    _save("6a_erosion.png", eroded)
    _print_done("Erosion (SE rect 5×5)", OUTPUT_DIR)

    # Dilation: memperbesar objek, mengisi lubang kecil
    dilated  = cv2.dilate(binary, se_rect, iterations=1)
    _save("6a_dilation.png", dilated)
    _print_done("Dilation (SE rect 5×5)", OUTPUT_DIR)

    # Opening: erosion → dilation (hapus noise kecil, pertahankan bentuk)
    opened   = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  se_ellip)
    _save("6b_opening.png", opened)
    _print_done("Opening  (erosion → dilation)", OUTPUT_DIR)

    # Closing: dilation → erosion (tutup lubang kecil, hubungkan objek)
    closed   = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, se_ellip)
    _save("6b_closing.png", closed)
    _print_done("Closing  (dilation → erosion)", OUTPUT_DIR)

    # --- 6b. Morfologi lanjutan ---
    # Gradient morfologi = dilation - erosion (tepi objek)
    grad_morph = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, se_rect)
    _save("6c_gradient_morph.png", grad_morph)
    _print_done("Morphological gradient (tepi)", OUTPUT_DIR)

    # Top-hat = original - opening (bagian lebih terang dari sekitar)
    tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT,   se_ellip)
    _save("6c_tophat.png", tophat)
    _print_done("Top-hat transform", OUTPUT_DIR)

    # Black-hat = closing - original (bagian lebih gelap dari sekitar)
    blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, se_ellip)
    _save("6c_blackhat.png", blackhat)
    _print_done("Black-hat transform", OUTPUT_DIR)

    # --- 6c. Morfologi grayscale ---
    se_g = np.ones((5, 5), dtype=np.uint8)
    erode_g  = cv2.erode(gray, se_g)
    dilate_g = cv2.dilate(gray, se_g)
    _save("6d_erosi_grayscale.png", erode_g)
    _save("6d_dilasi_grayscale.png", dilate_g)
    _print_done("Morfologi grayscale (erosion & dilation)", OUTPUT_DIR)

    # --- 6d. Skeletonization via thinning berulang ---
    skel = np.zeros_like(binary)
    temp = binary.copy()
    kernel_sk = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(30):
        e    = cv2.erode(temp, kernel_sk)
        d    = cv2.dilate(e, kernel_sk)
        skel = cv2.bitwise_or(skel, cv2.subtract(temp, d))
        temp = e.copy()
        if cv2.countNonZero(temp) == 0:
            break
    _save("6e_skeleton.png", skel)
    _print_done("Skeletonization (thinning)", OUTPUT_DIR)

    return binary, eroded, dilated


# ══════════════════════════════════════════════════════════════════════════════
#  BAB 7 — SEGMENTASI CITRA
# ══════════════════════════════════════════════════════════════════════════════

def bab7_segmentasi(gray, img_bgr):
    """
    Topik: Partisi citra menjadi region yang bermakna.
    Pendekatan: thresholding, region-based, clustering, watershed.
    """
    _print_section("BAB 7 · SEGMENTASI CITRA")

    # --- 7a. Global Thresholding ---
    _, global_t = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    _save("7a_global_thresh.png", global_t)
    _print_done("Global thresholding (T=127)", OUTPUT_DIR)

    # Iterative optimal thresholding (manual)
    T = gray.mean()
    for _ in range(100):
        g1 = gray[gray >= T].mean() if (gray >= T).any() else T
        g2 = gray[gray <  T].mean() if (gray <  T).any() else T
        T_new = (g1 + g2) / 2
        if abs(T_new - T) < 0.5:
            break
        T = T_new
    _, iter_t = cv2.threshold(gray, int(T), 255, cv2.THRESH_BINARY)
    _save("7a_iterative_thresh.png", iter_t)
    _print_done(f"Iterative thresholding (T={T:.1f})", OUTPUT_DIR)

    # --- 7b. K-Means Segmentation (NumPy manual) ---
    K = 3
    pix_f = img_bgr.reshape(-1, 3).astype(np.float64)
    np.random.seed(42)
    centers = pix_f[np.random.choice(len(pix_f), K, replace=False)]

    for _ in range(20):
        # Hitung jarak setiap piksel ke setiap centroid
        dists  = np.linalg.norm(pix_f[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        # Update centroid
        new_centers = np.array([
            pix_f[labels == k].mean(axis=0) if (labels == k).any() else centers[k]
            for k in range(K)
        ])
        if np.allclose(centers, new_centers, atol=0.5):
            break
        centers = new_centers

    km_img = centers[labels].reshape(img_bgr.shape).astype(np.uint8)
    _save("7b_kmeans_numpy.png", km_img)
    _print_done(f"K-Means segmentation (K={K})", OUTPUT_DIR)

    # --- 7c. Region Growing (seed-based) ---
    H, W   = gray.shape
    seed   = (H // 2, W // 2)
    thresh_rg = 20
    region = np.zeros((H, W), dtype=np.uint8)
    visited = np.zeros((H, W), dtype=bool)
    stack   = [seed]
    seed_val = int(gray[seed])

    while stack:
        y, x = stack.pop()
        if visited[y, x]:
            continue
        visited[y, x] = True
        if abs(int(gray[y, x]) - seed_val) <= thresh_rg:
            region[y, x] = 255
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
                    stack.append((ny, nx))

    _save("7c_region_growing.png", region)
    _print_done("Region growing (seed = pusat)", OUTPUT_DIR)

    # --- 7d. Watershed segmentation ---
    _, sure_fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sure_fg    = cv2.erode(sure_fg, np.ones((3,3), np.uint8), iterations=3)
    sure_bg    = cv2.dilate(sure_fg, np.ones((3,3), np.uint8), iterations=3)
    unknown    = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers    = markers + 1
    markers[unknown == 255] = 0

    img_ws = img_bgr.copy()
    cv2.watershed(img_ws, markers)
    img_ws[markers == -1] = [0, 0, 255]   # batas merah
    _save("7d_watershed.png", img_ws)
    _print_done("Watershed segmentation", OUTPUT_DIR)

    # --- 7e. Connected Components Labeling ---
    n_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(
        sure_fg, connectivity=8
    )
    # Warnai setiap komponen berbeda
    cc_color = np.zeros((*gray.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    for lbl in range(1, n_labels):
        color = rng.integers(80, 255, size=3).tolist()
        cc_color[labels_cc == lbl] = color
    _save("7e_connected_components.png", cc_color)
    _print_done(f"Connected components ({n_labels-1} objek)", OUTPUT_DIR)

    return km_img, region


# ══════════════════════════════════════════════════════════════════════════════
#  BAB 8 — TRANSFORMASI GEOMETRI
# ══════════════════════════════════════════════════════════════════════════════

def bab8_transformasi_geometri(img_bgr):
    """
    Topik: Transformasi koordinat spasial citra.
    Termasuk translasi, rotasi, skala, shear, dan transformasi perspektif.
    """
    _print_section("BAB 8 · TRANSFORMASI GEOMETRI")

    H, W  = img_bgr.shape[:2]
    cx, cy = W / 2.0, H / 2.0

    # --- 8a. Translasi ---
    tx, ty = 50, 30
    M_trans = np.float32([[1, 0, tx],
                           [0, 1, ty]])
    translated = cv2.warpAffine(img_bgr, M_trans, (W, H))
    _save("8a_translasi.png", translated)
    _print_done(f"Translasi (tx={tx}, ty={ty})", OUTPUT_DIR)

    # --- 8b. Rotasi ---
    M_rot = cv2.getRotationMatrix2D((cx, cy), 45, 1.0)
    rotated = cv2.warpAffine(img_bgr, M_rot, (W, H))
    _save("8b_rotasi_45.png", rotated)
    _print_done("Rotasi 45° (center)", OUTPUT_DIR)

    # --- 8c. Skala (scaling) ---
    scaled_up   = cv2.resize(img_bgr, (W*2, H*2), interpolation=cv2.INTER_LINEAR)
    scaled_down = cv2.resize(img_bgr, (W//2, H//2), interpolation=cv2.INTER_AREA)
    _save("8c_skala_2x.png", scaled_up)
    _save("8c_skala_half.png", scaled_down)
    _print_done("Skala 2× dan 0.5×", OUTPUT_DIR)

    # --- 8d. Shear transform ---
    shear_x = 0.3
    M_shear = np.float32([[1, shear_x, 0],
                           [0, 1,      0]])
    sheared = cv2.warpAffine(img_bgr, M_shear, (W, H))
    _save("8d_shear.png", sheared)
    _print_done("Shear transform (sx=0.3)", OUTPUT_DIR)

    # --- 8e. Flip (pencerminan) ---
    flip_h = cv2.flip(img_bgr, 1)   # horizontal
    flip_v = cv2.flip(img_bgr, 0)   # vertikal
    flip_b = cv2.flip(img_bgr, -1)  # keduanya
    _save("8e_flip_horizontal.png", flip_h)
    _save("8e_flip_vertikal.png", flip_v)
    _save("8e_flip_keduanya.png", flip_b)
    _print_done("Flip (horizontal, vertikal, keduanya)", OUTPUT_DIR)

    # --- 8f. Perspektif transform ---
    pts_src = np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])
    offset  = W // 5
    pts_dst = np.float32([[offset,0],[W-1-offset,0],[W-1,H-1],[0,H-1]])
    M_persp = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped  = cv2.warpPerspective(img_bgr, M_persp, (W, H))
    _save("8f_perspektif.png", warped)
    _print_done("Perspektif transform", OUTPUT_DIR)

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
#  BAB 9 — RESTORASI & REDUKSI NOISE
# ══════════════════════════════════════════════════════════════════════════════

def bab9_restorasi_noise(gray):
    """
    Topik: Pemodelan degradasi citra dan teknik restorasi.
    Model degradasi: g = H(f) + η
    """
    _print_section("BAB 9 · RESTORASI & REDUKSI NOISE")

    # --- 9a. Tambahkan berbagai tipe noise ---
    # Gaussian noise
    noise_gauss = gray.astype(np.float64) + np.random.normal(0, 25, gray.shape)
    noisy_g = np.clip(noise_gauss, 0, 255).astype(np.uint8)
    _save("9a_noise_gaussian.png", noisy_g)
    _print_done("Tambah Gaussian noise (σ=25)", OUTPUT_DIR)

    # Salt & Pepper noise
    noisy_sp = gray.copy()
    rng = np.random.default_rng(0)
    mask_s = rng.random(gray.shape) < 0.03
    mask_p = rng.random(gray.shape) < 0.03
    noisy_sp[mask_s] = 255
    noisy_sp[mask_p] = 0
    _save("9a_noise_salt_pepper.png", noisy_sp)
    _print_done("Tambah Salt & Pepper noise (3%)", OUTPUT_DIR)

    # Speckle noise
    noise_speckle = gray + gray * np.random.randn(*gray.shape) * 0.1
    noisy_speck   = np.clip(noise_speckle, 0, 255).astype(np.uint8)
    _save("9a_noise_speckle.png", noisy_speck)
    _print_done("Tambah Speckle noise", OUTPUT_DIR)

    # --- 9b. Reduksi noise ---
    # Mean filter (vs Gaussian noise)
    k_mean = np.ones((3,3), np.float64) / 9
    denoised_mean = np.clip(ndimage.convolve(noisy_g.astype(np.float64), k_mean), 0, 255).astype(np.uint8)
    _save("9b_denoised_mean.png", denoised_mean)
    _print_done("Mean filter (Gaussian noise)", OUTPUT_DIR)

    # Median filter (terbaik untuk salt & pepper)
    denoised_median = ndimage.median_filter(noisy_sp, size=3)
    _save("9b_denoised_median.png", denoised_median.astype(np.uint8))
    _print_done("Median filter (salt & pepper)", OUTPUT_DIR)

    # Bilateral filter (jaga tepi sambil denoise)
    denoised_bilateral = cv2.bilateralFilter(noisy_g, d=9, sigmaColor=75, sigmaSpace=75)
    _save("9b_denoised_bilateral.png", denoised_bilateral)
    _print_done("Bilateral filter (edge-preserving)", OUTPUT_DIR)

    # Wiener filter (optimal MSE — via frekuensi)
    F     = np.fft.fft2(noisy_g.astype(np.float64))
    K     = 0.01   # rasio noise-to-signal
    H_blur= np.fft.fft2(np.ones((5,5)) / 25, s=gray.shape)
    Hc    = np.conj(H_blur)
    wiener_f = (Hc / (np.abs(H_blur)**2 + K)) * F
    wiener_img = np.abs(np.fft.ifft2(wiener_f))
    _save("9b_denoised_wiener.png", np.clip(wiener_img, 0, 255).astype(np.uint8))
    _print_done("Wiener filter (frekuensi domain)", OUTPUT_DIR)

    # --- 9c. Blind deblurring sederhana — inverse filter ---
    # Simulasi motion blur
    size_mb = 15
    kernel_mb = np.zeros((size_mb, size_mb))
    kernel_mb[size_mb//2, :] = 1.0 / size_mb
    blurred = cv2.filter2D(gray, -1, kernel_mb)
    _save("9c_motion_blurred.png", blurred)

    # Inverse filter di domain frekuensi
    F_blur  = np.fft.fft2(blurred.astype(np.float64))
    H_mb    = np.fft.fft2(kernel_mb, s=gray.shape)
    eps     = 0.01
    inv_f   = F_blur / (H_mb + eps)
    inv_img = np.abs(np.fft.ifft2(inv_f))
    _save("9c_inverse_filter.png", np.clip(inv_img, 0, 255).astype(np.uint8))
    _print_done("Motion blur + inverse filter", OUTPUT_DIR)

    return noisy_g, noisy_sp, denoised_median


# ══════════════════════════════════════════════════════════════════════════════
#  BAB 10 — KOMPRESI CITRA
# ══════════════════════════════════════════════════════════════════════════════

def bab10_kompresi(gray):
    """
    Topik: Reduksi ukuran data citra dengan / tanpa kehilangan informasi.
    Lossless: RLE, Huffman. Lossy: JPEG-like (DCT + kuantisasi).
    """
    _print_section("BAB 10 · KOMPRESI CITRA")

    # --- 10a. Run-Length Encoding (RLE) — lossless ---
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

    # Terapkan RLE pada citra biner
    _, biner_rle = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    vals, cnts  = rle_encode(biner_rle)
    rle_size    = len(vals) + len(cnts)
    orig_size   = biner_rle.size
    rle_ratio   = rle_size / orig_size

    reconstructed_rle = rle_decode(vals, cnts, biner_rle.shape)
    _save("10a_rle_rekonstruksi.png", reconstructed_rle)
    print(f"  RLE original  : {orig_size:,} nilai")
    print(f"  RLE encoded   : {rle_size:,} pasang (ratio: {rle_ratio:.3f})")
    _print_done("RLE encode + decode (lossless)", OUTPUT_DIR)

    # --- 10b. JPEG-like lossy compression (DCT + quantization) ---
    # Matriks kuantisasi luminance standar JPEG
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

    results_quality = {}
    for q_scale in [1.0, 4.0, 16.0]:     # 1=HQ, 4=medium, 16=LQ
        recon = np.zeros((H8, W8), dtype=np.float64)
        Q_scaled = Q_luma * q_scale

        for y in range(0, H8, 8):
            for x in range(0, W8, 8):
                blok = gray8[y:y+8, x:x+8]
                D    = dct(dct(blok, axis=0, norm='ortho'), axis=1, norm='ortho')
                Dq   = np.round(D / Q_scaled) * Q_scaled
                R    = idct(idct(Dq, axis=1, norm='ortho'), axis=0, norm='ortho')
                recon[y:y+8, x:x+8] = R

        recon_img = np.clip(recon + 128, 0, 255).astype(np.uint8)
        _save(f"10b_jpeg_q{int(q_scale)}.png", recon_img)

        # Hitung PSNR
        mse  = np.mean((gray[:H8,:W8].astype(float) - recon_img.astype(float))**2)
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
        results_quality[q_scale] = psnr
        print(f"  JPEG Q×{q_scale:4.0f}  PSNR = {psnr:.2f} dB")

    _print_done("JPEG-like compression (3 kualitas)", OUTPUT_DIR)

    # --- 10c. Predictive coding (DPCM) — lossless ---
    # Encode perbedaan antar piksel (delta coding)
    pred  = np.zeros_like(gray, dtype=np.int16)
    pred[:, 0] = gray[:, 0]
    pred[:, 1:] = gray[:, 1:].astype(np.int16) - gray[:, :-1].astype(np.int16)

    recon_pred = np.cumsum(pred.astype(np.int16), axis=1)
    recon_pred_u8 = np.clip(recon_pred, 0, 255).astype(np.uint8)
    _save("10c_dpcm_delta.png", np.clip(pred + 128, 0, 255).astype(np.uint8))
    _save("10c_dpcm_rekonstruksi.png", recon_pred_u8)
    _print_done("DPCM predictive coding (delta)", OUTPUT_DIR)

    return recon_img


# ══════════════════════════════════════════════════════════════════════════════
#  BAB 11 — DETEKSI FITUR & ANALISIS OBJEK
# ══════════════════════════════════════════════════════════════════════════════

def bab11_fitur_objek(gray, img_bgr):
    """
    Topik: Ekstraksi deskriptor dan deteksi fitur geometris.
    Digunakan dalam pengenalan pola, computer vision, AR.
    """
    _print_section("BAB 11 · DETEKSI FITUR & ANALISIS OBJEK")

    # --- 11a. Harris Corner Detection ---
    gray_f   = np.float32(gray)
    harris   = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)
    harris_n = cv2.normalize(harris, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    img_harris = img_bgr.copy()
    img_harris[harris > 0.01 * harris.max()] = [0, 0, 255]
    _save("11a_harris_corners.png", img_harris)
    corner_count = np.sum(harris > 0.01 * harris.max())
    _print_done(f"Harris corners ({corner_count} titik)", OUTPUT_DIR)

    # --- 11b. SIFT (Scale-Invariant Feature Transform) ---
    try:
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        img_sift = cv2.drawKeypoints(img_bgr, kp, None,
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        _save("11b_sift_keypoints.png", img_sift)
        _print_done(f"SIFT keypoints ({len(kp)} titik)", OUTPUT_DIR)
    except Exception:
        _print_done("SIFT (tidak tersedia di build ini)", OUTPUT_DIR)

    # --- 11c. ORB (Oriented FAST + Rotated BRIEF) ---
    orb = cv2.ORB_create(nfeatures=500)
    kp_orb, des_orb = orb.detectAndCompute(gray, None)
    img_orb = cv2.drawKeypoints(img_bgr, kp_orb, None, color=(0, 255, 0))
    _save("11c_orb_keypoints.png", img_orb)
    _print_done(f"ORB keypoints ({len(kp_orb)} titik)", OUTPUT_DIR)

    # --- 11d. Deteksi kontur dan momen ---
    _, biner = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(biner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_contour = img_bgr.copy()
    cv2.drawContours(img_contour, contours, -1, (0, 255, 0), 1)

    # Hitung momen untuk setiap kontur
    for i, cnt in enumerate(contours[:5]):
        M_cnt = cv2.moments(cnt)
        if M_cnt["m00"] > 0:
            cx_cnt = int(M_cnt["m10"] / M_cnt["m00"])
            cy_cnt = int(M_cnt["m01"] / M_cnt["m00"])
            cv2.circle(img_contour, (cx_cnt, cy_cnt), 5, (0, 0, 255), -1)

    _save("11d_kontur_momen.png", img_contour)
    _print_done(f"Kontur + momen ({len(contours)} kontur)", OUTPUT_DIR)

    # --- 11e. Deskriptor bentuk (shape descriptors) ---
    props = []
    for cnt in contours[:10]:
        area     = cv2.contourArea(cnt)
        perim    = cv2.arcLength(cnt, True)
        if perim > 0 and area > 100:
            circularity = 4 * np.pi * area / (perim ** 2)
            props.append((area, perim, circularity))

    if props:
        print(f"  Contour (5 terbesar):")
        props_sorted = sorted(props, key=lambda x: -x[0])[:5]
        for a, p, c in props_sorted:
            print(f"    area={a:.0f}  perimeter={p:.1f}  circularity={c:.3f}")

    # --- 11f. Hough Transform — deteksi garis dan lingkaran ---
    edges   = cv2.Canny(gray, 50, 150)
    lines   = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                               minLineLength=30, maxLineGap=10)
    img_lines = img_bgr.copy()
    if lines is not None:
        for line in lines[:20]:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_lines, (x1,y1), (x2,y2), (0, 255, 0), 1)
    _save("11f_hough_lines.png", img_lines)
    _print_done(f"Hough lines ({len(lines) if lines is not None else 0} garis)", OUTPUT_DIR)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                param1=50, param2=30, minRadius=10, maxRadius=100)
    img_circles = img_bgr.copy()
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for (cx_c, cy_c, r) in circles[:10]:
            cv2.circle(img_circles, (cx_c, cy_c), r, (0, 255, 0), 2)
            cv2.circle(img_circles, (cx_c, cy_c), 2, (0, 0, 255), 3)
    _save("11f_hough_circles.png", img_circles)
    _print_done(f"Hough circles ({len(circles) if circles is not None else 0} lingkaran)", OUTPUT_DIR)

    return img_contour


# ══════════════════════════════════════════════════════════════════════════════
#  BAB 12 — PENGOLAHAN CITRA WARNA LANJUTAN
# ══════════════════════════════════════════════════════════════════════════════

def bab12_warna_lanjutan(img_bgr):
    """
    Topik: Manipulasi lanjutan dalam berbagai ruang warna.
    Color constancy, color transfer, pseudo-coloring, serta analisis warna.
    """
    _print_section("BAB 12 · PENGOLAHAN WARNA LANJUTAN")

    # --- 12a. Histogram equalization per channel (HSV) ---
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])   # equalize hanya channel V
    eq_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    _save("12a_color_equalized.png", eq_color)
    _print_done("Color equalization (channel V)", OUTPUT_DIR)

    # --- 12b. Color transfer (von Reinhard method) ---
    # Transfer warna rata-rata dan std dari sumber ke target dalam LAB
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    # Buat target artificial (citra yang sama di-blur sangat kuat sebagai "referensi")
    target  = cv2.GaussianBlur(img_bgr, (99, 99), 30)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float64)

    result_lab = img_lab.copy()
    for c in range(3):
        src_mean, src_std = img_lab[:,:,c].mean(), img_lab[:,:,c].std()
        tgt_mean, tgt_std = tgt_lab[:,:,c].mean(), tgt_lab[:,:,c].std()
        if src_std > 0:
            result_lab[:,:,c] = (img_lab[:,:,c] - src_mean) * (tgt_std / src_std) + tgt_mean

    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    color_transferred = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    _save("12b_color_transfer.png", color_transferred)
    _print_done("Color transfer (Reinhard, LAB)", OUTPUT_DIR)

    # --- 12c. Pseudo-coloring (colormap pada grayscale) ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    for cmap_name, cmap_id in [("jet", cv2.COLORMAP_JET),
                                 ("hot", cv2.COLORMAP_HOT),
                                 ("viridis", cv2.COLORMAP_VIRIDIS)]:
        pseudo = cv2.applyColorMap(gray, cmap_id)
        _save(f"12c_pseudocolor_{cmap_name}.png", pseudo)
    _print_done("Pseudo-coloring (jet, hot, viridis)", OUTPUT_DIR)

    # --- 12d. Isolasi warna (HSV masking) ---
    hsv2  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Isolasi warna merah (dua range karena H melingkar)
    mask1 = cv2.inRange(hsv2, np.array([0,120,70]),   np.array([10,255,255]))
    mask2 = cv2.inRange(hsv2, np.array([170,120,70]), np.array([180,255,255]))
    mask_red = cv2.bitwise_or(mask1, mask2)
    isolated = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_red)
    _save("12d_isolasi_merah.png", isolated)
    _print_done("Isolasi warna merah (HSV masking)", OUTPUT_DIR)

    # --- 12e. White balance (Gray World Assumption) ---
    img_f = img_bgr.astype(np.float64)
    mean_b, mean_g, mean_r = img_f[:,:,0].mean(), img_f[:,:,1].mean(), img_f[:,:,2].mean()
    gray_world = (mean_b + mean_g + mean_r) / 3
    wb = img_f.copy()
    wb[:,:,0] *= gray_world / (mean_b + 1e-6)
    wb[:,:,1] *= gray_world / (mean_g + 1e-6)
    wb[:,:,2] *= gray_world / (mean_r + 1e-6)
    _save("12e_white_balance.png", np.clip(wb, 0, 255).astype(np.uint8))
    _print_done("White balance (Gray World)", OUTPUT_DIR)

    return eq_color, color_transferred


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — INPUT PENGGUNA & ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  PENGOLAHAN CITRA DIGITAL — Implementasi Lengkap NumPy  ║")
    print("║  12 Bab · 60+ Teknik · Output otomatis ke folder        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Input citra dari pengguna
    path = input("\nMasukkan path citra (atau tekan Enter untuk citra uji sintetis): ").strip().strip('"')

    if path and os.path.exists(path):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print("[ERROR] Tidak dapat membaca citra.")
            sys.exit(1)
        print(f"[OK] Citra dimuat: {img_bgr.shape[1]}×{img_bgr.shape[0]} px")
    else:
        # Buat citra sintetis jika tidak ada input
        print("[INFO] Membuat citra sintetis 256×256 untuk demonstrasi ...")
        img_bgr = np.zeros((256, 256, 3), dtype=np.uint8)
        for y in range(256):
            for x in range(256):
                img_bgr[y, x, 2] = x                           # R
                img_bgr[y, x, 1] = y                           # G
                img_bgr[y, x, 0] = (x + y) // 2               # B
        # Tambah lingkaran dan persegi untuk kepentingan morfologi & segmentasi
        cv2.circle(img_bgr, (128, 128), 60, (255, 200, 100), -1)
        cv2.circle(img_bgr, (64,   64), 30, (100, 255, 150), -1)
        cv2.circle(img_bgr, (192,  64), 25, (150, 100, 255), -1)
        cv2.rectangle(img_bgr, (30, 150), (100, 220), (200, 100, 200), -1)
        cv2.rectangle(img_bgr, (160,150), (230, 210), (100, 200, 200), -1)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "_input_sintetis.png"), img_bgr)
        print(f"[OK] Citra sintetis dibuat dan disimpan.")

    _save("_input_asli.png", img_bgr)

    # Jalankan semua bab
    t_total = time.perf_counter()

    gray, hsv, lab                     = bab1_akuisisi_representasi(img_bgr)
    contrast                           = bab2_transformasi_intensitas(gray)
    gray_eq                            = bab3_histogram(gray)
    gauss_f, sobel                     = bab4_filter_spasial(gray, img_bgr)
    img_lp, img_hp                     = bab5_domain_frekuensi(gray)
    binary, eroded, dilated            = bab6_morfologi(gray)
    km_img, region                     = bab7_segmentasi(gray, img_bgr)
    rotated, warped                    = bab8_transformasi_geometri(img_bgr)
    noisy_g, noisy_sp, denoised        = bab9_restorasi_noise(gray)
    jpeg_recon                         = bab10_kompresi(gray)
    img_contour                        = bab11_fitur_objek(gray, img_bgr)
    eq_color, color_tf                 = bab12_warna_lanjutan(img_bgr)

    elapsed_total = time.perf_counter() - t_total

    # Hitung output
    n_files = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])

    print(f"\n{'═'*60}")
    print(f"  SELESAI")
    print(f"  Total waktu   : {elapsed_total:.2f} detik")
    print(f"  File output   : {n_files} citra PNG")
    print(f"  Folder output : ./{OUTPUT_DIR}/")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
