"""
image_processor.py
==================
Modul logika pemrosesan citra untuk Aplikasi Web PCD.
Berisi kelas-kelas terpisah untuk setiap domain tugas akhir:
  - FrequencyDomainProcessor  : FFT, LPF, HPF (Ideal & Butterworth)
  - GeometricTransformProcessor: Translasi, Rotasi, Skala + Interpolasi
  - CompressionAnalysisProcessor: DCT Blok 8×8, Kuantisasi, PSNR/MSE

Semua fungsi mengembalikan dict hasil agar mudah dikonsumsi oleh app.py.
BGR ↔ RGB dikelola di sini agar app.py cukup berurusan dengan RGB.
"""

import time
import numpy as np
import cv2
from scipy.fft import dct, idct
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Helper Umum
# ─────────────────────────────────────────────────────────────────────────────

def _to_gray(img_rgb: np.ndarray) -> np.ndarray:
    """Konversi RGB uint8 → grayscale uint8."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def _ensure_uint8(arr: np.ndarray) -> np.ndarray:
    """Clip dan konversi array float ke uint8."""
    return np.clip(arr, 0, 255).astype(np.uint8)


def _normalize_log_magnitude(spectrum: np.ndarray) -> np.ndarray:
    """Skala logaritmik untuk spektrum FFT → uint8 untuk visualisasi."""
    mag = np.log1p(np.abs(spectrum))
    mag_n = (mag / mag.max() * 255)
    return _ensure_uint8(mag_n)


def _apply_colormap(gray_img: np.ndarray, cmap=cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Terapkan colormap OpenCV → kembalikan RGB."""
    colored_bgr = cv2.applyColorMap(gray_img, cmap)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def create_sample_image(size: int = 256) -> np.ndarray:
    """
    Versi Super Cepat (Vectorized) - Aman dan Instan hingga ukuran 2048+
    """
    # 1. Buat skala warna 0-255 langsung disesuaikan dengan 'size'
    skala_warna = np.linspace(0, 255, size).astype(np.uint8)
    mesh_x, mesh_y = np.meshgrid(skala_warna, skala_warna)
    
    # 2. Alokasikan matriks gambar
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # 3. Masukkan nilai secara grosiran (Tanpa loop 'for' Python)
    img[:, :, 0] = mesh_x                              # R
    img[:, :, 1] = mesh_y                              # G
    img[:, :, 2] = (mesh_x.astype(np.uint16) + mesh_y.astype(np.uint16)) // 2 # B (Convert ke uint16 dulu saat ditambah agar tidak overflow sebelum dibagi)

    # 4. Gambar objek geometri (Otomatis menyesuaikan skala 'size')
    cv2.circle(img, (size // 2, size // 2), size // 4, (255, 200, 80), -1)
    cv2.circle(img, (size // 4, size // 4), size // 8, (80, 255, 150), -1)
    cv2.circle(img, (3 * size // 4, size // 4), size // 10, (150, 80, 255), -1)
    cv2.rectangle(img, (20, size // 2 + 20), (size // 3, size - 20), (200, 80, 200), -1)
    cv2.rectangle(img, (size // 2 + 20, size // 2 + 20), (size - 20, size - 20), (80, 200, 200), -1)
    
    return img


# ─────────────────────────────────────────────────────────────────────────────
# MENU 1 — Transformasi Domain Frekuensi (FFT)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FFTResult:
    original_rgb: np.ndarray
    magnitude_vis: np.ndarray        # Spektrum magnitude colormap (RGB)
    magnitude_gray: np.ndarray       # Spektrum magnitude grayscale
    phase_vis: np.ndarray            # Spektrum fase (RGB)
    filtered_rgb: np.ndarray         # Citra hasil filter
    reconstructed_rgb: np.ndarray    # Citra rekonstruksi dari IFFT
    filter_mask_vis: np.ndarray      # Visualisasi topeng filter (RGB)
    filter_type: str
    cutoff: int
    elapsed_ms: float
    psnr_db: float
    filter_name: str


class FrequencyDomainProcessor:
    """Pemrosesan citra dalam domain frekuensi menggunakan FFT 2D."""

    # Matriks kuantisasi standar JPEG (referensi)
    _JPEG_Q_MATRIX = np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99],
    ], dtype=np.float64)

    def _build_distance_map(self, H: int, W: int) -> np.ndarray:
        crow, ccol = H // 2, W // 2
        Y, X = np.ogrid[:H, :W]
        return np.sqrt((X - ccol) ** 2 + (Y - crow) ** 2)

    def _ideal_lpf_mask(self, D: np.ndarray, cutoff: int) -> np.ndarray:
        return (D <= cutoff).astype(np.float64)

    def _ideal_hpf_mask(self, D: np.ndarray, cutoff: int) -> np.ndarray:
        return (D > cutoff).astype(np.float64)

    def _butterworth_lpf_mask(self, D: np.ndarray, cutoff: int, order: int = 2) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            return 1.0 / (1.0 + (D / (cutoff + 1e-6)) ** (2 * order))

    def _butterworth_hpf_mask(self, D: np.ndarray, cutoff: int, order: int = 2) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            bw_lp = 1.0 / (1.0 + (D / (cutoff + 1e-6)) ** (2 * order))
            return 1.0 - bw_lp

    def _compute_psnr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        orig_f = original.astype(np.float64)
        rec_f  = reconstructed.astype(np.float64)
        mse = np.mean((orig_f - rec_f) ** 2)
        if mse < 1e-10:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))

    def process(
        self,
        img_rgb: np.ndarray,
        cutoff: int = 30,
        filter_type: str = "LPF Ideal",
        butterworth_order: int = 2,
    ) -> FFTResult:
        """
        Jalankan FFT 2D, aplikasikan filter, dan rekonstruksi.

        Parameters
        ----------
        img_rgb       : Citra RGB uint8
        cutoff        : Radius cutoff frequency (piksel dari pusat)
        filter_type   : Salah satu dari ["LPF Ideal","HPF Ideal","LPF Butterworth","HPF Butterworth"]
        butterworth_order: Orde filter Butterworth
        """
        t0 = time.perf_counter()
        try:
            gray = _to_gray(img_rgb).astype(np.float64)
            H, W = gray.shape

            # 1. FFT 2D + fftshift (DC ke tengah)
            f_complex = np.fft.fft2(gray)
            f_shifted = np.fft.fftshift(f_complex)

            # 2. Spektrum magnitude & fase
            mag_gray = _normalize_log_magnitude(f_shifted)
            mag_vis  = _apply_colormap(mag_gray, cv2.COLORMAP_INFERNO)
            phase_raw = np.angle(f_shifted)
            phase_n   = _ensure_uint8((phase_raw + np.pi) / (2 * np.pi) * 255)
            phase_vis = _apply_colormap(phase_n, cv2.COLORMAP_TURBO)

            # 3. Bangun mask filter
            D = self._build_distance_map(H, W)
            if filter_type == "LPF Ideal":
                mask = self._ideal_lpf_mask(D, cutoff)
                fname = f"Low-Pass Filter Ideal (r={cutoff})"
            elif filter_type == "HPF Ideal":
                mask = self._ideal_hpf_mask(D, cutoff)
                fname = f"High-Pass Filter Ideal (r={cutoff})"
            elif filter_type == "LPF Butterworth":
                mask = self._butterworth_lpf_mask(D, cutoff, butterworth_order)
                fname = f"Low-Pass Butterworth (r={cutoff}, n={butterworth_order})"
            else:  # HPF Butterworth
                mask = self._butterworth_hpf_mask(D, cutoff, butterworth_order)
                fname = f"High-Pass Butterworth (r={cutoff}, n={butterworth_order})"

            # Visualisasi mask filter (diberi colormap)
            mask_vis_gray = _ensure_uint8(mask * 255)
            mask_vis = _apply_colormap(mask_vis_gray, cv2.COLORMAP_MAGMA)

            # 4. Aplikasikan filter di domain frekuensi
            f_filtered = f_shifted * mask

            # 5. Rekonstruksi via IFFT
            f_ishift     = np.fft.ifftshift(f_filtered)
            img_back     = np.abs(np.fft.ifft2(f_ishift))
            filtered_gray = _ensure_uint8(img_back)

            # Konversi grayscale → RGB untuk tampilan
            filtered_rgb      = cv2.cvtColor(filtered_gray, cv2.COLOR_GRAY2RGB)
            reconstructed_rgb = filtered_rgb.copy()

            # 6. Hitung PSNR
            orig_gray_u8 = _to_gray(img_rgb)
            psnr = self._compute_psnr(orig_gray_u8, filtered_gray)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            return FFTResult(
                original_rgb=img_rgb,
                magnitude_vis=mag_vis,
                magnitude_gray=mag_gray,
                phase_vis=phase_vis,
                filtered_rgb=filtered_rgb,
                reconstructed_rgb=reconstructed_rgb,
                filter_mask_vis=mask_vis,
                filter_type=filter_type,
                cutoff=cutoff,
                elapsed_ms=elapsed_ms,
                psnr_db=psnr,
                filter_name=fname,
            )
        except Exception as e:
            raise RuntimeError(f"[FFT] Kesalahan pemrosesan: {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# MENU 2 — Transformasi Geometri Kompleks
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeometricResult:
    original_rgb: np.ndarray
    translated_rgb: np.ndarray
    rotated_rgb: np.ndarray
    scaled_rgb: np.ndarray
    interp_results: dict            # {nama_interp: np.ndarray RGB}
    tx: int
    ty: int
    angle: float
    scale_x: float
    scale_y: float
    interp_method: str
    elapsed_ms: float


class GeometricTransformProcessor:
    """Transformasi geometri: Translasi, Rotasi, Skala + komparasi Interpolasi."""

    INTERP_MAP = {
        "Nearest Neighbor": cv2.INTER_NEAREST,
        "Bilinear":          cv2.INTER_LINEAR,
        "Bicubic":           cv2.INTER_CUBIC,
        "Lanczos4":          cv2.INTER_LANCZOS4,
        "Area (Downscale)":  cv2.INTER_AREA,
    }

    def _translate(self, img: np.ndarray, tx: int, ty: int, interp: int) -> np.ndarray:
        H, W = img.shape[:2]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(img, M, (W, H), flags=interp,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def _rotate(self, img: np.ndarray, angle: float, interp: int) -> np.ndarray:
        H, W = img.shape[:2]
        cx, cy = W / 2.0, H / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        return cv2.warpAffine(img, M, (W, H), flags=interp,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def _scale(self, img: np.ndarray, sx: float, sy: float, interp: int) -> np.ndarray:
        H, W = img.shape[:2]
        new_W = max(1, int(W * sx))
        new_H = max(1, int(H * sy))
        scaled = cv2.resize(img, (new_W, new_H), interpolation=interp)
        # Kembalikan ke ukuran asli dengan padding/crop agar mudah dibandingkan
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        ph = min(new_H, H)
        pw = min(new_W, W)
        canvas[:ph, :pw] = scaled[:ph, :pw]
        return canvas

    def process(
        self,
        img_rgb: np.ndarray,
        tx: int = 40,
        ty: int = 20,
        angle: float = 45.0,
        scale_x: float = 1.5,
        scale_y: float = 1.5,
        interp_method: str = "Bilinear",
    ) -> GeometricResult:
        """
        Terapkan transformasi geometri + bandingkan semua metode interpolasi.

        Parameters
        ----------
        img_rgb      : Citra RGB uint8
        tx, ty       : Vektor translasi (piksel)
        angle        : Sudut rotasi (derajat, positif = searah jarum jam di OpenCV)
        scale_x/y    : Faktor skala horizontal/vertikal
        interp_method: Metode interpolasi utama yang digunakan untuk translasi & rotasi
        """
        t0 = time.perf_counter()
        try:
            interp_cv = self.INTERP_MAP.get(interp_method, cv2.INTER_LINEAR)

            translated  = self._translate(img_rgb, tx, ty, interp_cv)
            rotated     = self._rotate(img_rgb, angle, interp_cv)
            scaled      = self._scale(img_rgb, scale_x, scale_y, interp_cv)

            # Perbandingan interpolasi: downscale lalu upscale ke ukuran asli
            H, W = img_rgb.shape[:2]
            small = cv2.resize(img_rgb, (max(1, W // 4), max(1, H // 4)))
            interp_results = {}
            for name, flag in self.INTERP_MAP.items():
                up = cv2.resize(small, (W, H), interpolation=flag)
                interp_results[name] = up

            elapsed_ms = (time.perf_counter() - t0) * 1000
            return GeometricResult(
                original_rgb=img_rgb,
                translated_rgb=translated,
                rotated_rgb=rotated,
                scaled_rgb=scaled,
                interp_results=interp_results,
                tx=tx,
                ty=ty,
                angle=angle,
                scale_x=scale_x,
                scale_y=scale_y,
                interp_method=interp_method,
                elapsed_ms=elapsed_ms,
            )
        except Exception as e:
            raise RuntimeError(f"[Geometri] Kesalahan pemrosesan: {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# MENU 3 — Analisis Kompresi & Kuantisasi (DCT)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CompressionResult:
    original_rgb: np.ndarray
    compressed_rgb: np.ndarray
    dct_vis_rgb: np.ndarray          # Koefisien DCT (colormap)
    quantized_dct_vis_rgb: np.ndarray
    quality_factor: int
    mse: float
    psnr_db: float
    compression_ratio: float         # Estimasi rasio kompresi
    nonzero_ratio: float             # Persentase koefisien non-zero
    elapsed_ms: float
    quality_assessment: str          # Teks analisis otomatis


# Matriks kuantisasi luminansi standar JPEG
_JPEG_Q_LUM = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99],
], dtype=np.float64)


class CompressionAnalysisProcessor:
    """Kompresi JPEG-like: DCT 8×8, kuantisasi, rekonstruksi, dan metrik."""

    def _build_quant_matrix(self, quality_factor: int) -> np.ndarray:
        """
        Bangun matriks kuantisasi berdasarkan quality_factor (1–95).
        Rumus skalar JPEG-standard:
          S = 5000/Q  jika Q < 50, else S = 200 - 2Q
          Qmat = clip(floor((S * Qbase + 50) / 100), 1, 255)
        """
        q = max(1, min(quality_factor, 95))
        s = 5000 / q if q < 50 else 200 - 2 * q
        qmat = np.floor((s * _JPEG_Q_LUM + 50) / 100)
        return np.clip(qmat, 1, 255)

    def _dct2(self, block: np.ndarray) -> np.ndarray:
        return dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

    def _idct2(self, block: np.ndarray) -> np.ndarray:
        return idct(idct(block, axis=1, norm='ortho'), axis=0, norm='ortho')

    def _compute_metrics(self, orig: np.ndarray, compressed: np.ndarray):
        orig_f = orig.astype(np.float64)
        comp_f = compressed.astype(np.float64)
        mse = np.mean((orig_f - comp_f) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse + 1e-10))
        return float(mse), float(psnr)

    def _quality_assessment(self, psnr: float, quality_factor: int) -> str:
        """Hasilkan teks analisis trade-off kualitas secara otomatis."""
        if psnr >= 45:
            level = "Sangat Tinggi"
            desc  = ("PSNR ≥ 45 dB menunjukkan kualitas rekonstruksi yang hampir sempurna. "
                     "Perbedaan visual antara citra asli dan hasil kompresi nyaris tidak terdeteksi "
                     "oleh mata manusia. Cocok untuk keperluan arsip dan medis.")
        elif psnr >= 38:
            level = "Tinggi"
            desc  = ("PSNR 38–45 dB menunjukkan kualitas baik. "
                     "Artefak kompresi (blocking, ringing) mungkin terlihat pada tepi tajam "
                     "bila dilihat dari dekat, namun secara umum kualitas visual masih memuaskan.")
        elif psnr >= 30:
            level = "Sedang"
            desc  = ("PSNR 30–38 dB adalah rentang tipikal kompresi JPEG standar. "
                     "Artefak blok 8×8 mulai terlihat terutama pada area transisi warna. "
                     "Trade-off: ukuran file lebih kecil namun detail halus mulai hilang.")
        elif psnr >= 22:
            level = "Rendah"
            desc  = ("PSNR 22–30 dB menunjukkan kompresi agresif. "
                     "Artefak blok dan ringing jelas terlihat. "
                     "Informasi frekuensi tinggi (tepi, tekstur) banyak yang hilang karena "
                     "matriks kuantisasi yang besar menyebabkan banyak koefisien DCT di-zero-kan.")
        else:
            level = "Sangat Rendah"
            desc  = ("PSNR < 22 dB menandakan degradasi berat. "
                     "Hampir semua koefisien DCT bernilai nol kecuali komponen DC (rata-rata blok). "
                     "Citra tampak tersegmentasi menjadi blok-blok warna rata 8×8 piksel. "
                     "Tidak disarankan untuk konten visual apapun.")

        return (f"**Kualitas: {level}** | Quality Factor: {quality_factor} | PSNR: {psnr:.2f} dB\n\n"
                f"{desc}")

    def process(
        self,
        img_rgb: np.ndarray,
        quality_factor: int = 50,
    ) -> CompressionResult:
        """
        Kompresi JPEG-like dengan DCT blok 8×8.

        Parameters
        ----------
        img_rgb        : Citra RGB uint8
        quality_factor : Faktor kualitas 1 (terendah) – 95 (tertinggi)
        """
        t0 = time.perf_counter()
        try:
            gray = _to_gray(img_rgb).astype(np.float64)
            H, W = gray.shape

            # Potong ke kelipatan 8
            H8, W8 = (H // 8) * 8, (W // 8) * 8
            gray8 = gray[:H8, :W8]

            qmat = self._build_quant_matrix(quality_factor)

            # Array hasil
            dct_img   = np.zeros((H8, W8), dtype=np.float64)
            qdct_img  = np.zeros((H8, W8), dtype=np.float64)
            recon_img = np.zeros((H8, W8), dtype=np.float64)

            total_coeff   = 0
            nonzero_coeff = 0

            # Level-shift: geser ke [-128, 127]
            shifted = gray8 - 128.0

            for y in range(0, H8, 8):
                for x in range(0, W8, 8):
                    blok = shifted[y:y+8, x:x+8]

                    # DCT 2D
                    D = self._dct2(blok)
                    dct_img[y:y+8, x:x+8] = D

                    # Kuantisasi
                    Dq = np.round(D / qmat)
                    qdct_img[y:y+8, x:x+8] = Dq

                    # Hitung koefisien non-zero
                    total_coeff   += 64
                    nonzero_coeff += int(np.count_nonzero(Dq))

                    # De-kuantisasi
                    Ddeq = Dq * qmat

                    # IDCT 2D
                    R = self._idct2(Ddeq)
                    recon_img[y:y+8, x:x+8] = R

            # Kembalikan level-shift
            recon_gray = np.clip(recon_img + 128.0, 0, 255).astype(np.uint8)

            # Visualisasi koefisien DCT (skala log)
            dct_log     = np.log1p(np.abs(dct_img))
            dct_vis_u8  = _ensure_uint8(dct_log / dct_log.max() * 255)
            dct_vis_rgb = _apply_colormap(dct_vis_u8, cv2.COLORMAP_JET)

            qdct_log     = np.log1p(np.abs(qdct_img))
            qdct_max     = qdct_log.max()
            qdct_vis_u8  = _ensure_uint8(qdct_log / (qdct_max + 1e-10) * 255)
            qdct_vis_rgb = _apply_colormap(qdct_vis_u8, cv2.COLORMAP_HOT)

            # Konversi rekonstruksi ke RGB
            compressed_rgb = cv2.cvtColor(recon_gray, cv2.COLOR_GRAY2RGB)

            # Metrik
            orig_crop  = gray8.astype(np.uint8)
            mse, psnr  = self._compute_metrics(orig_crop, recon_gray)

            nonzero_ratio    = nonzero_coeff / total_coeff if total_coeff > 0 else 1.0
            compression_ratio = 1.0 / (nonzero_ratio + 1e-6)  # Estimasi sederhana

            quality_text = self._quality_assessment(psnr, quality_factor)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            return CompressionResult(
                original_rgb=img_rgb,
                compressed_rgb=compressed_rgb,
                dct_vis_rgb=dct_vis_rgb,
                quantized_dct_vis_rgb=qdct_vis_rgb,
                quality_factor=quality_factor,
                mse=mse,
                psnr_db=psnr,
                compression_ratio=float(compression_ratio),
                nonzero_ratio=float(nonzero_ratio),
                elapsed_ms=elapsed_ms,
                quality_assessment=quality_text,
            )
        except Exception as e:
            raise RuntimeError(f"[DCT] Kesalahan pemrosesan: {e}") from e
