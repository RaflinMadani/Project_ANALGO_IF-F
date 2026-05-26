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
#  FUNGSI PERKALIAN MATRIKS — BRUTE FORCE
#  Menggantikan operator @ / np.matmul / np.dot dari versi numpy
#  Kompleksitas: O(n³) — triple nested loop
# ══════════════════════════════════════════════════════════════════════════════

def matmul_brute_force(A, B):
    """
    Perkalian dua matriks menggunakan triple nested loop (Brute Force).

    Algoritma:
        Untuk setiap baris i dari A dan kolom j dari B:
            C[i][j] = Σ A[i][k] * B[k][j]  untuk k = 0..K-1

    Kompleksitas Waktu : O(n × m × p)
                         n = baris A, m = kolom B, p = kolom A = baris B
    Kompleksitas Ruang : O(n × m) untuk matriks hasil C

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
    n, p = A.shape
    m    = B.shape[1]

    C = np.zeros((n, m), dtype=np.float64)

    # Triple nested loop — definisi klasik perkalian matriks
    for i in range(n):
        for j in range(m):
            for k in range(p):
                C[i][j] += A[i][k] * B[k][j]

    return C


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


# ── Helper: warpAffine manual dengan Brute Force ─────────────────────────────

def _warp_affine_brute(img, M2x3):
    """
    Terapkan transformasi affine per piksel menggunakan inverse mapping.
    Perkalian matriks M @ [x, y, 1]^T dilakukan dengan matmul_brute_force.

    Kompleksitas: O(H × W × 6)  — 6 operasi multiply-add per piksel
    """
    H, W = img.shape[:2]
    out  = np.zeros_like(img)

    # Invers matriks affine 2×2 (komponen rotasi/skala)
    a, b, tx = M2x3[0]
    c, d, ty = M2x3[1]
    det = a * d - b * c
    if abs(det) < 1e-10:
        return out
    inv_det = 1.0 / det
    Ainv = np.array([[ d * inv_det, -b * inv_det],
                      [-c * inv_det,  a * inv_det]], dtype=np.float64)
    t_vec = np.array([[tx], [ty]], dtype=np.float64)

    for y_dst in range(H):
        for x_dst in range(W):
            # Koordinat sumber: [x_src, y_src]^T = A_inv @ ([x_dst,y_dst]^T - t)
            dst_vec = np.array([[x_dst], [y_dst]], dtype=np.float64)
            src = matmul_brute_force(Ainv, dst_vec - t_vec)
            ix  = int(round(float(src[0, 0])))
            iy  = int(round(float(src[1, 0])))
            if 0 <= ix < W and 0 <= iy < H:
                out[y_dst, x_dst] = img[iy, ix]

    return out


def _warp_perspective_brute(img, M3x3):
    """
    Terapkan transformasi perspektif per piksel dengan inverse mapping.
    Perkalian matriks M_inv @ [x, y, 1]^T memakai matmul_brute_force.

    Kompleksitas: O(H × W × 9)  — 9 operasi multiply-add per piksel
    """
    H, W   = img.shape[:2]
    out    = np.zeros_like(img)
    M_inv  = np.linalg.inv(M3x3.astype(np.float64))

    for y_dst in range(H):
        for x_dst in range(W):
            pt_dst = np.array([[x_dst], [y_dst], [1.0]], dtype=np.float64)
            pt_src = matmul_brute_force(M_inv, pt_dst)
            w      = pt_src[2, 0]
            if abs(w) < 1e-10:
                continue
            ix = int(round(pt_src[0, 0] / w))
            iy = int(round(pt_src[1, 0] / w))
            if 0 <= ix < W and 0 <= iy < H:
                out[y_dst, x_dst] = img[iy, ix]

    return out


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

    # --- 5c. DCT 2D — menggunakan matmul_brute_force ---
    # DCT 2D blok = D @ blok @ D.T
    # Kedua perkalian matriks dilakukan dengan Brute Force O(n³)
    H8 = (H // 8) * 8
    W8 = (W // 8) * 8
    gray8 = gray[:H8, :W8].astype(np.float64) - 128   # level shift

    D8       = _dct_matrix(8)      # matriks basis DCT 8×8
    D8T      = D8.T                # transpose untuk IDCT

    dct_img  = np.zeros_like(gray8)
    idct_img = np.zeros_like(gray8)

    for y in range(0, H8, 8):
        for x in range(0, W8, 8):
            blok = gray8[y:y+8, x:x+8]

            # DCT 2D: D @ blok @ D.T  — dua kali matmul_brute_force
            temp = matmul_brute_force(D8,  blok)   # O(8³) = O(512)
            Dblk = matmul_brute_force(temp, D8T)   # O(8³) = O(512)

            dct_img[y:y+8, x:x+8] = Dblk

            # Kuantisasi sederhana (faktor Q=10) → simulasi kompresi
            Q  = np.ones((8, 8)) * 10
            Dq = np.round(Dblk / Q) * Q

            # IDCT 2D: D.T @ Dq @ D  — dua kali matmul_brute_force
            temp2 = matmul_brute_force(D8T, Dq)   # O(8³) = O(512)
            R     = matmul_brute_force(temp2, D8)  # O(8³) = O(512)

            idct_img[y:y+8, x:x+8] = R

    dct_vis = np.log(1 + np.abs(dct_img))
    dct_vis = (dct_vis / dct_vis.max() * 255).astype(np.uint8)
    _save("5c_dct_koefisien.png", dct_vis)

    recon = np.clip(idct_img + 128, 0, 255).astype(np.uint8)
    _save("5c_dct_rekonstruksi.png", recon)
    _print_done("DCT 2D blok 8×8 + rekonstruksi", OUTPUT_DIR)

    return img_lp, img_hp

# ══════════════════════════════════════════════════════════════════════════════
#  BAB 8 — TRANSFORMASI GEOMETRI
# ══════════════════════════════════════════════════════════════════════════════

def bab8_transformasi_geometri(img_bgr):
    """
    Topik: Transformasi koordinat spasial citra.
    Termasuk translasi, rotasi, skala, shear, dan transformasi perspektif.
    Perkalian matriks affine/perspektif menggunakan matmul_brute_force.
    """
    _print_section("BAB 8 · TRANSFORMASI GEOMETRI")

    H, W  = img_bgr.shape[:2]
    cx, cy = W / 2.0, H / 2.0

    # --- 8a. Translasi ---
    tx, ty = 50, 30
    M_trans = np.float64([[1, 0, tx],
                           [0, 1, ty]])
    # Translasi menggunakan inverse mapping + matmul_brute_force per piksel
    translated = _warp_affine_brute(img_bgr, M_trans)
    _save("8a_translasi.png", translated)
    _print_done(f"Translasi (tx={tx}, ty={ty})", OUTPUT_DIR)

    # --- 8b. Rotasi ---
    rad   = np.deg2rad(45)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    M_rot = np.float64([
        [cos_a, -sin_a, (1 - cos_a)*cx + sin_a*cy],
        [sin_a,  cos_a, (1 - cos_a)*cy - sin_a*cx],
    ])
    rotated = _warp_affine_brute(img_bgr, M_rot)
    _save("8b_rotasi_45.png", rotated)
    _print_done("Rotasi 45° (center)", OUTPUT_DIR)

    # --- 8c. Skala (scaling) ---
    # Skala tidak melibatkan perkalian matriks non-trivial — tetap cv2.resize
    scaled_up   = cv2.resize(img_bgr, (W*2, H*2), interpolation=cv2.INTER_LINEAR)
    scaled_down = cv2.resize(img_bgr, (W//2, H//2), interpolation=cv2.INTER_AREA)
    _save("8c_skala_2x.png", scaled_up)
    _save("8c_skala_half.png", scaled_down)
    _print_done("Skala 2× dan 0.5×", OUTPUT_DIR)

    # --- 8d. Shear transform ---
    shear_x = 0.3
    M_shear = np.float64([[1, shear_x, 0],
                           [0, 1,      0]])
    sheared = _warp_affine_brute(img_bgr, M_shear)
    _save("8d_shear.png", sheared)
    _print_done("Shear transform (sx=0.3)", OUTPUT_DIR)

    # --- 8e. Flip (pencerminan) ---
    flip_h = cv2.flip(img_bgr, 1)
    flip_v = cv2.flip(img_bgr, 0)
    flip_b = cv2.flip(img_bgr, -1)
    _save("8e_flip_horizontal.png", flip_h)
    _save("8e_flip_vertikal.png", flip_v)
    _save("8e_flip_keduanya.png", flip_b)
    _print_done("Flip (horizontal, vertikal, keduanya)", OUTPUT_DIR)

    # --- 8f. Perspektif transform ---
    pts_src = np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])
    offset  = W // 5
    pts_dst = np.float32([[offset,0],[W-1-offset,0],[W-1,H-1],[0,H-1]])
    M_persp = cv2.getPerspectiveTransform(pts_src, pts_dst)
    # Perspektif: perkalian matriks 3×3 × [x,y,1]^T via matmul_brute_force
    warped  = _warp_perspective_brute(img_bgr, M_persp)
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
#  BAB 10 — KOMPRESI CITRA
# ══════════════════════════════════════════════════════════════════════════════

def bab10_kompresi(gray):
    """
    Topik: Reduksi ukuran data citra dengan / tanpa kehilangan informasi.
    Lossless: RLE, Huffman. Lossy: JPEG-like (DCT + kuantisasi).
    Perkalian matriks DCT menggunakan matmul_brute_force.
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

    # --- 10b. JPEG-like lossy compression — DCT via matmul_brute_force ---
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

    results_quality = {}
    for q_scale in [1.0, 4.0, 16.0]:
        recon    = np.zeros((H8, W8), dtype=np.float64)
        Q_scaled = Q_luma * q_scale

        for y in range(0, H8, 8):
            for x in range(0, W8, 8):
                blok = gray8[y:y+8, x:x+8]

                # DCT 2D: D @ blok @ D.T  — matmul_brute_force
                temp = matmul_brute_force(D8,  blok)
                D_   = matmul_brute_force(temp, D8T)

                Dq   = np.round(D_ / Q_scaled) * Q_scaled

                # IDCT 2D: D.T @ Dq @ D  — matmul_brute_force
                temp2 = matmul_brute_force(D8T, Dq)
                R     = matmul_brute_force(temp2, D8)

                recon[y:y+8, x:x+8] = R

        recon_img = np.clip(recon + 128, 0, 255).astype(np.uint8)
        _save(f"10b_jpeg_q{int(q_scale)}.png", recon_img)

        mse  = np.mean((gray[:H8,:W8].astype(float) - recon_img.astype(float))**2)
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
        results_quality[q_scale] = psnr
        print(f"  JPEG Q×{q_scale:4.0f}  PSNR = {psnr:.2f} dB")

    _print_done("JPEG-like compression (3 kualitas)", OUTPUT_DIR)

    # --- 10c. Predictive coding (DPCM) — lossless ---
    pred  = np.zeros_like(gray, dtype=np.int16)
    pred[:, 0] = gray[:, 0]
    pred[:, 1:] = gray[:, 1:].astype(np.int16) - gray[:, :-1].astype(np.int16)

    recon_pred    = np.cumsum(pred.astype(np.int16), axis=1)
    recon_pred_u8 = np.clip(recon_pred, 0, 255).astype(np.uint8)
    _save("10c_dpcm_delta.png", np.clip(pred + 128, 0, 255).astype(np.uint8))
    _save("10c_dpcm_rekonstruksi.png", recon_pred_u8)
    _print_done("DPCM predictive coding (delta)", OUTPUT_DIR)

    return recon_img

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — INPUT PENGGUNA & ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔═══════════════════════════╗")
    print("║                           ║")
    print("║  PENGOLAHAN CITRA DIGITAL ║")  
    print("║                           ║")
    print("╚═══════════════════════════╝")

    path = input("\nMasukkan path citra (atau tekan Enter untuk citra uji sintetis): ").strip().strip('"')

    if path and os.path.exists(path):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print("[ERROR] Tidak dapat membaca citra.")
            sys.exit(1)
        print(f"[OK] Citra dimuat: {img_bgr.shape[1]}×{img_bgr.shape[0]} px")
    else:
        print("[INFO] Membuat citra sintetis 256×256 untuk demonstrasi ...")
        img_bgr = np.zeros((256, 256, 3), dtype=np.uint8)
        for y in range(256):
            for x in range(256):
                img_bgr[y, x, 2] = x
                img_bgr[y, x, 1] = y
                img_bgr[y, x, 0] = (x + y) // 2
        cv2.circle(img_bgr, (128, 128), 60, (255, 200, 100), -1)
        cv2.circle(img_bgr, (64,   64), 30, (100, 255, 150), -1)
        cv2.circle(img_bgr, (192,  64), 25, (150, 100, 255), -1)
        cv2.rectangle(img_bgr, (30, 150), (100, 220), (200, 100, 200), -1)
        cv2.rectangle(img_bgr, (160,150), (230, 210), (100, 200, 200), -1)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "_input_sintetis.png"), img_bgr)
        print(f"[OK] Citra sintetis dibuat dan disimpan.")

    _save("_input_asli.png", img_bgr)

    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    t_total = time.perf_counter()

    img_lp, img_hp  = bab5_domain_frekuensi(gray)
    rotated, warped = bab8_transformasi_geometri(img_bgr)
    jpeg_recon      = bab10_kompresi(gray)

    elapsed_total = time.perf_counter() - t_total
    n_files = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])

    print(f"\n{'═'*60}")
    print(f"  SELESAI  [Metode: BRUTE FORCE]")
    print(f"  Total waktu   : {elapsed_total:.2f} detik")
    print(f"  File output   : {n_files} citra PNG")
    print(f"  Folder output : ./{OUTPUT_DIR}/")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
