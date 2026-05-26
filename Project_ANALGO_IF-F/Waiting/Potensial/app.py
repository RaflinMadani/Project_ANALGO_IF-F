"""
app.py
======
Aplikasi Web Interaktif — Repositori Proyek Akhir Mata Kuliah
Pengolahan Citra Digital (PCD)

Struktur:
  - SessionStateManager : pengelola state Streamlit terpusat
  - render_menu_fft()   : Tab 1 — Domain Frekuensi (FFT)
  - render_menu_geo()   : Tab 2 — Transformasi Geometri
  - render_menu_dct()   : Tab 3 — Kompresi & Kuantisasi
  - main()              : Entry point aplikasi

Jalankan dengan:
  streamlit run app.py
"""

import time
import io
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from image_processor import (
    FrequencyDomainProcessor,
    GeometricTransformProcessor,
    CompressionAnalysisProcessor,
    create_sample_image,
    _to_gray,
    FFTResult,
    GeometricResult,
    CompressionResult,
)

# ─────────────────────────────────────────────────────────────────────────────
# Konfigurasi Halaman
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PCD Dashboard | Repositori Tugas Akhir",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS Kustom — Dark Mode Professional
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Root variables */
:root {
    --primary:      #6C63FF;
    --primary-dark: #4A3FDB;
    --accent:       #00D4AA;
    --warn:         #FF6B6B;
    --bg-card:      #1E1E2E;
    --bg-widget:    #252535;
    --border:       rgba(108,99,255,0.25);
    --text-muted:   #A0A0C0;
    --radius:       12px;
}

/* Global font */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main container */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13131F 0%, #1A1A2E 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--primary) !important;
}

/* Header Hero */
.hero-header {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: "";
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(108,99,255,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.hero-title span { color: var(--primary); }
.hero-subtitle {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin: 0;
}

/* Metric Cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.metric-unit {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.2rem;
}

/* Image caption style */
.img-caption {
    font-size: 0.78rem;
    color: var(--text-muted);
    text-align: center;
    margin-top: 0.3rem;
    font-style: italic;
}

/* Section title */
.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #FFFFFF;
    border-left: 3px solid var(--primary);
    padding-left: 0.7rem;
    margin: 1rem 0 0.7rem 0;
}

/* Execution time badge */
.exec-badge {
    display: inline-block;
    background: rgba(0,212,170,0.15);
    border: 1px solid rgba(0,212,170,0.35);
    color: var(--accent);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
}

/* Alert / trade-off box */
.tradeoff-box {
    background: rgba(108,99,255,0.08);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
}

/* Warning box */
.warn-box {
    background: rgba(255,107,107,0.08);
    border: 1px solid rgba(255,107,107,0.3);
    border-radius: var(--radius);
    padding: 0.8rem 1rem;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text-muted);
    font-weight: 500;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: var(--primary) !important;
    border-color: var(--primary) !important;
    color: white !important;
}

/* Divider */
.pcd-divider {
    height: 1px;
    background: var(--border);
    margin: 1rem 0;
}

/* Info chip */
.chip {
    display: inline-block;
    background: rgba(108,99,255,0.15);
    color: #A09FFF;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 1px;
}

/* Scrollbar */
::-webkit-scrollbar       { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #13131F; }
::-webkit-scrollbar-thumb { background: #3A3A5C; border-radius: 3px; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State Manager
# ─────────────────────────────────────────────────────────────────────────────

class SessionStateManager:
    """Pengelola terpusat untuk st.session_state — tanpa variabel global."""

    KEYS = {
        "img_rgb":          None,   # np.ndarray RGB uint8
        "img_name":         "—",
        "img_loaded":       False,
        "fft_result":       None,
        "geo_result":       None,
        "dct_result":       None,
    }

    @staticmethod
    def init():
        for key, default in SessionStateManager.KEYS.items():
            if key not in st.session_state:
                st.session_state[key] = default

    @staticmethod
    def set_image(img_rgb: np.ndarray, name: str = "uploaded"):
        st.session_state.img_rgb    = img_rgb
        st.session_state.img_name   = name
        st.session_state.img_loaded = True
        # Reset hasil saat citra baru dimuat
        st.session_state.fft_result = None
        st.session_state.geo_result = None
        st.session_state.dct_result = None

    @staticmethod
    def get_image() -> tuple[bool, np.ndarray | None]:
        return st.session_state.img_loaded, st.session_state.img_rgb


# ─────────────────────────────────────────────────────────────────────────────
# Helper UI
# ─────────────────────────────────────────────────────────────────────────────

def _display_image(col, img_rgb: np.ndarray, caption: str, use_container: bool = True):
    """Tampilkan citra RGB dengan caption yang konsisten."""
    col.image(img_rgb, caption=caption, use_container_width=use_container)


def _exec_badge(elapsed_ms: float) -> str:
    return f'<span class="exec-badge">⏱ {elapsed_ms:.1f} ms</span>'


def _section_title(text: str):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def _metric_card(label: str, value: str, unit: str = ""):
    st.markdown(
        f"""<div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
              <div class="metric-unit">{unit}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def _plot_spectrum(mag_gray: np.ndarray, title: str = "Spektrum FFT") -> plt.Figure:
    """Buat plot spektrum magnitude FFT dengan matplotlib."""
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#1E1E2E")
    ax.set_facecolor("#1E1E2E")
    im = ax.imshow(mag_gray, cmap="inferno", aspect="auto")
    plt.colorbar(im, ax=ax, label="Log Magnitude")
    ax.set_title(title, color="white", fontsize=10, pad=8)
    ax.set_xlabel("Frekuensi Horizontal (u)", color="#A0A0C0", fontsize=8)
    ax.set_ylabel("Frekuensi Vertikal (v)",   color="#A0A0C0", fontsize=8)
    ax.tick_params(colors="#A0A0C0", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#3A3A5C")
    fig.tight_layout()
    return fig


def _plot_dct_heatmap(dct_vis: np.ndarray, title: str) -> plt.Figure:
    """Plot koefisien DCT sebagai heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#1E1E2E")
    ax.set_facecolor("#1E1E2E")
    im = ax.imshow(dct_vis, cmap="jet", aspect="auto")
    plt.colorbar(im, ax=ax, label="Log |DCT|")
    ax.set_title(title, color="white", fontsize=10, pad=8)
    ax.set_xlabel("Blok — arah X", color="#A0A0C0", fontsize=8)
    ax.set_ylabel("Blok — arah Y", color="#A0A0C0", fontsize=8)
    ax.tick_params(colors="#A0A0C0", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#3A3A5C")
    fig.tight_layout()
    return fig


def _plot_psnr_gauge(psnr: float) -> plt.Figure:
    """Visualisasi gauge PSNR."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 2.5), facecolor="#1E1E2E")
    ax.set_facecolor("#1E1E2E")
    thresholds = [22, 30, 38, 45, 60]
    colors      = ["#FF4444", "#FF8800", "#FFD700", "#44FF88", "#00CCFF"]
    labels      = ["Buruk", "Rendah", "Sedang", "Baik", "Sangat Baik"]
    bar_starts  = [0,  22, 30, 38, 45]
    bar_widths  = [22, 8,  8,  7,  15]

    for i, (start, width, color) in enumerate(zip(bar_starts, bar_widths, colors)):
        ax.barh(0, width, left=start, color=color, alpha=0.3, height=0.6)
        ax.text(start + width / 2, -0.55, labels[i], ha="center",
                va="top", color=color, fontsize=7)

    clipped = min(psnr, 60)
    ax.axvline(clipped, color="white", linewidth=2.5, linestyle="--", zorder=5)
    ax.text(clipped, 0.4, f"{psnr:.1f} dB", ha="center", va="bottom",
            color="white", fontsize=9, fontweight="bold")

    ax.set_xlim(0, 60)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("PSNR (dB)", color="#A0A0C0", fontsize=8)
    ax.set_yticks([])
    ax.tick_params(colors="#A0A0C0", labelsize=7)
    ax.set_title("Indikator Kualitas PSNR", color="white", fontsize=9, pad=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#3A3A5C")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Input Citra
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    """Render sidebar input citra dan informasi repositori."""
    with st.sidebar:
        st.markdown(
            """<div style="text-align:center; padding: 1rem 0 0.5rem 0;">
                 <div style="font-size:2.5rem;">🔬</div>
                 <div style="font-size:1rem; font-weight:700; color:#6C63FF; margin:0.3rem 0;">
                     PCD Dashboard
                 </div>
                 <div style="font-size:0.72rem; color:#A0A0C0;">
                     Repositori Tugas Akhir
                 </div>
               </div>""",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Input Citra ──────────────────────────────────────────────
        st.markdown("### 📂 Input Citra")
        input_mode = st.radio(
            "Pilih sumber citra:",
            ["Unggah Gambar Sendiri", "Gunakan Gambar Sampel"],
            index=1,
            label_visibility="collapsed",
        )

        if input_mode == "Unggah Gambar Sendiri":
            uploaded = st.file_uploader(
                "Pilih file JPG / JPEG / PNG",
                type=["jpg", "jpeg", "png"],
                help="Maksimum ukuran file: 200 MB",
            )
            if uploaded is not None:
                try:
                    pil_img = Image.open(uploaded).convert("RGB")
                    img_rgb = np.array(pil_img, dtype=np.uint8)
                    SessionStateManager.set_image(img_rgb, name=uploaded.name)
                    st.success(f"✅ Citra dimuat: **{uploaded.name}**")
                except Exception as e:
                    st.error(f"❌ Gagal membaca citra: {e}")

        else:  # Gambar Sampel
            sample_size = st.select_slider(
                "Ukuran citra sampel (piksel):",
                options=[128, 256, 512, 1024, 2048],
                value=256,
            )
            if st.button("🔄 Muat Ulang Gambar Sampel", use_container_width=True):
                img_rgb = create_sample_image(sample_size)
                SessionStateManager.set_image(img_rgb, name=f"Sintetis_{sample_size}px")
                st.success(f"✅ Gambar sampel {sample_size}×{sample_size} dimuat.")

            # Auto-load jika belum ada gambar
            loaded, _ = SessionStateManager.get_image()
            if not loaded:
                img_rgb = create_sample_image(sample_size)
                SessionStateManager.set_image(img_rgb, name=f"Sintetis_{sample_size}px")

        # ── Info Citra Aktif ─────────────────────────────────────────
        loaded, img_rgb = SessionStateManager.get_image()
        if loaded and img_rgb is not None:
            st.divider()
            st.markdown("### 📊 Info Citra Aktif")
            H, W = img_rgb.shape[:2]
            gray_info = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

            st.markdown(
                f"""
                | Parameter     | Nilai |
                |--------------|-------|
                | **Nama**     | {st.session_state.img_name} |
                | **Dimensi**  | {W} × {H} px |
                | **Channel**  | RGB (3) |
                | **Min**      | {img_rgb.min()} |
                | **Max**      | {img_rgb.max()} |
                | **Mean**     | {gray_info.mean():.1f} |
                | **Std Dev**  | {gray_info.std():.1f} |
                | **Ukuran**   | {img_rgb.nbytes / 1024:.1f} KB |
                """
            )
            st.image(img_rgb, caption="Preview", use_container_width=True)

        # ── Info Repositori ──────────────────────────────────────────
        st.divider()
        st.markdown(
            """<div style="font-size:0.75rem; color:#A0A0C0; line-height:1.7;">
                 <b style="color:#6C63FF;">📚 Tentang Aplikasi</b><br>
                 Repositori interaktif tugas akhir mata kuliah<br>
                 <b>Pengolahan Citra Digital (PCD)</b>.<br><br>
                 <b>Menu 1:</b> Domain Frekuensi (FFT)<br>
                 <b>Menu 2:</b> Transformasi Geometri<br>
                 <b>Menu 3:</b> Kompresi DCT & Kuantisasi<br><br>
                 Built with <b>Streamlit</b> + <b>OpenCV</b> + <b>NumPy</b>
               </div>""",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MENU 1 — Transformasi Domain Frekuensi (FFT)
# ─────────────────────────────────────────────────────────────────────────────

def render_menu_fft(img_rgb: np.ndarray):
    """Tab 1: FFT 2D, filter frekuensi, rekonstruksi."""

    st.markdown('<div class="hero-header"><p class="hero-title">📡 Menu 1: <span>Domain Frekuensi (FFT 2D)</span></p><p class="hero-subtitle">Fast Fourier Transform — Low-Pass &amp; High-Pass Filter (Ideal &amp; Butterworth)</p></div>', unsafe_allow_html=True)

    # ── Panel Kontrol ────────────────────────────────────────────────
    with st.expander("⚙️ Parameter Filter", expanded=True):
        col_p1, col_p2, col_p3 = st.columns([2, 1, 1])
        with col_p1:
            filter_type = st.selectbox(
                "Jenis Filter:",
                ["LPF Ideal", "HPF Ideal", "LPF Butterworth", "HPF Butterworth"],
                help="LPF = Low-Pass Filter, HPF = High-Pass Filter",
            )
        with col_p2:
            cutoff = st.slider(
                "Cutoff Frequency (r):",
                min_value=1, max_value=min(img_rgb.shape[:2]) // 2,
                value=30,
                help="Radius masker dalam piksel dari pusat spektrum",
            )
        with col_p3:
            bw_order = st.slider(
                "Orde Butterworth (n):",
                min_value=1, max_value=10, value=2,
                disabled="Butterworth" not in filter_type,
                help="Hanya berlaku untuk filter Butterworth",
            )

        run_fft = st.button("🚀 Jalankan FFT", type="primary", use_container_width=True)

    # ── Eksekusi ─────────────────────────────────────────────────────
    if run_fft or st.session_state.fft_result is not None:
        if run_fft:
            processor = FrequencyDomainProcessor()
            with st.spinner("⚙️ Menghitung FFT 2D dan mengaplikasikan filter…"):
                try:
                    result: FFTResult = processor.process(
                        img_rgb,
                        cutoff=cutoff,
                        filter_type=filter_type,
                        butterworth_order=bw_order,
                    )
                    st.session_state.fft_result = result
                except RuntimeError as e:
                    st.error(str(e))
                    return

        result: FFTResult = st.session_state.fft_result
        if result is None:
            return

        # ── Metrik Ringkas ───────────────────────────────────────────
        st.markdown(
            f'<div style="margin-bottom:0.5rem;">{_exec_badge(result.elapsed_ms)} &nbsp; '
            f'<span class="chip">filter: {result.filter_name}</span></div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            _metric_card("PSNR", f"{result.psnr_db:.2f}", "dB")
        with m2:
            _metric_card("Cutoff Freq", str(result.cutoff), "piksel")
        with m3:
            H, W = img_rgb.shape[:2]
            _metric_card("Resolusi", f"{W}×{H}", "piksel")
        with m4:
            _metric_card("Eksekusi", f"{result.elapsed_ms:.1f}", "ms")

        st.markdown('<div class="pcd-divider"></div>', unsafe_allow_html=True)

        # ── Baris 1: Citra Asli vs Hasil Filter ─────────────────────
        _section_title("Perbandingan Citra Asli vs Hasil Filter")
        col_a, col_b = st.columns(2)
        _display_image(col_a, result.original_rgb,  "📷 Citra Asli (Domain Spasial)")
        _display_image(col_b, result.filtered_rgb,  f"🔧 Citra Hasil {result.filter_name}")

        st.markdown('<div class="pcd-divider"></div>', unsafe_allow_html=True)

        # ── Baris 2: Spektrum FFT ─────────────────────────────────
        _section_title("Spektrum Domain Frekuensi")
        col_c, col_d, col_e = st.columns(3)

        with col_c:
            fig_spec = _plot_spectrum(result.magnitude_gray, "Spektrum Magnitude (Log Scale)")
            st.pyplot(fig_spec, use_container_width=True)
            st.markdown('<p class="img-caption">DC (frekuensi nol) berada di tengah. Titik terang = energi dominan.</p>', unsafe_allow_html=True)
            plt.close(fig_spec)

        with col_d:
            _display_image(st, result.phase_vis, "🌈 Spektrum Fase (Colormap Turbo)")

        with col_e:
            _display_image(st, result.filter_mask_vis, f"🎭 Topeng Filter — {result.filter_type}")

        st.markdown('<div class="pcd-divider"></div>', unsafe_allow_html=True)

        # ── Penjelasan Teoritis ──────────────────────────────────────
        _section_title("📖 Penjelasan Teoritis")
        is_lpf = "LPF" in result.filter_type
        is_bw  = "Butterworth" in result.filter_type

        with st.expander("Klik untuk membaca penjelasan", expanded=False):
            st.markdown(f"""
**Transformasi Fourier 2D** mengubah citra dari domain spasial ke domain frekuensi:

$$F(u,v) = \\sum_{{x=0}}^{{M-1}} \\sum_{{y=0}}^{{N-1}} f(x,y) \\cdot e^{{-j2\\pi\\left(\\frac{{ux}}{{M}}+\\frac{{vy}}{{N}}\\right)}}$$

- **Frekuensi rendah** (dekat pusat spektrum): komponen warna rata dan kontur besar.
- **Frekuensi tinggi** (jauh dari pusat): tepi tajam, detail halus, dan noise.

**Filter yang aktif: `{result.filter_name}`**

{"**Low-Pass Filter** meloloskan frekuensi rendah (radius ≤ cutoff) dan menahan frekuensi tinggi → efek blur/smoothing." if is_lpf else "**High-Pass Filter** meloloskan frekuensi tinggi (radius > cutoff) dan menahan frekuensi rendah → efek penajaman tepi."}

{"**Butterworth** menggunakan fungsi transfer H(u,v) = 1 / [1 + (D/D₀)^(2n)] sehingga transisi halus (tanpa artefak ringing seperti filter Ideal)." if is_bw else "**Filter Ideal** menggunakan masker biner: nilai 1 jika D ≤ cutoff, 0 jika sebaliknya. Transisi langsung menyebabkan artefak *ringing* (efek Gibbs)."}
""")

    else:
        st.info("🔵 Atur parameter di atas lalu klik **Jalankan FFT** untuk memulai analisis.", icon="ℹ️")


# ─────────────────────────────────────────────────────────────────────────────
# MENU 2 — Transformasi Geometri Kompleks
# ─────────────────────────────────────────────────────────────────────────────

def render_menu_geo(img_rgb: np.ndarray):
    """Tab 2: Translasi, Rotasi, Skala, komparasi Interpolasi."""

    st.markdown('<div class="hero-header"><p class="hero-title">📐 Menu 2: <span>Transformasi Geometri</span></p><p class="hero-subtitle">Translasi • Rotasi • Skala — Komparasi Metode Interpolasi</p></div>', unsafe_allow_html=True)

    # ── Panel Kontrol ────────────────────────────────────────────────
    with st.expander("⚙️ Parameter Transformasi", expanded=True):
        H_img, W_img = img_rgb.shape[:2]

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            tx = st.slider("Translasi X (tx) — piksel:", -W_img // 2, W_img // 2, 40, step=5)
        with col_t2:
            ty = st.slider("Translasi Y (ty) — piksel:", -H_img // 2, H_img // 2, 20, step=5)

        col_r, col_s1, col_s2 = st.columns(3)
        with col_r:
            angle = st.slider("Rotasi (derajat):", -180, 180, 45, step=5,
                               help="Positif = searah jarum jam (konvensi OpenCV)")
        with col_s1:
            scale_x = st.slider("Skala X (sx):", 0.25, 3.0, 1.5, step=0.25)
        with col_s2:
            scale_y = st.slider("Skala Y (sy):", 0.25, 3.0, 1.5, step=0.25)

        interp_method = st.selectbox(
            "Metode Interpolasi (untuk Translasi & Rotasi):",
            list(GeometricTransformProcessor.INTERP_MAP.keys()),
            index=1,
            help="Interpolasi digunakan untuk memperkirakan nilai piksel pada koordinat non-integer.",
        )

        run_geo = st.button("🚀 Jalankan Transformasi", type="primary", use_container_width=True)

    # ── Eksekusi ─────────────────────────────────────────────────────
    if run_geo or st.session_state.geo_result is not None:
        if run_geo:
            processor = GeometricTransformProcessor()
            with st.spinner("⚙️ Menerapkan transformasi geometri…"):
                try:
                    result: GeometricResult = processor.process(
                        img_rgb,
                        tx=tx, ty=ty,
                        angle=float(angle),
                        scale_x=scale_x,
                        scale_y=scale_y,
                        interp_method=interp_method,
                    )
                    st.session_state.geo_result = result
                except RuntimeError as e:
                    st.error(str(e))
                    return

        result: GeometricResult = st.session_state.geo_result
        if result is None:
            return

        st.markdown(
            f'<div style="margin-bottom:0.5rem;">{_exec_badge(result.elapsed_ms)} &nbsp; '
            f'<span class="chip">interp: {result.interp_method}</span> '
            f'<span class="chip">angle: {result.angle}°</span> '
            f'<span class="chip">scale: {result.scale_x}×{result.scale_y}</span></div>',
            unsafe_allow_html=True,
        )

        # ── Tiga Transformasi Dasar ──────────────────────────────────
        _section_title("Tiga Transformasi Utama")
        col_o, col_tr, col_ro, col_sc = st.columns(4)
        _display_image(col_o,  result.original_rgb,   "📷 Asli")
        _display_image(col_tr, result.translated_rgb, f"↔ Translasi ({result.tx},{result.ty})")
        _display_image(col_ro, result.rotated_rgb,    f"🔄 Rotasi {result.angle}°")
        _display_image(col_sc, result.scaled_rgb,     f"🔍 Skala {result.scale_x}×{result.scale_y}")

        st.markdown('<div class="pcd-divider"></div>', unsafe_allow_html=True)

        # ── Komparasi Interpolasi ────────────────────────────────────
        _section_title("Komparasi Metode Interpolasi (Downscale ÷4 → Upscale ×4)")
        st.caption(
            "Setiap citra di-downscale 4× lalu di-upscale kembali ke ukuran asli "
            "menggunakan metode interpolasi yang berbeda. Perhatikan perbedaan detail dan artefak."
        )

        interp_items = list(result.interp_results.items())
        cols_interp = st.columns(len(interp_items))
        for col, (name, img) in zip(cols_interp, interp_items):
            with col:
                st.image(img, caption=name, use_container_width=True)
                # Hitung perbedaan MSE terhadap asli
                orig_g = _to_gray(result.original_rgb).astype(float)
                comp_g = _to_gray(img).astype(float)
                mse_i  = np.mean((orig_g - comp_g) ** 2)
                st.markdown(
                    f'<p class="img-caption">MSE: {mse_i:.1f}</p>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="pcd-divider"></div>', unsafe_allow_html=True)

        # ── Penjelasan Interpolasi ───────────────────────────────────
        _section_title("📖 Penjelasan Metode Interpolasi")
        with st.expander("Klik untuk membaca perbandingan metode"):
            tab_nn, tab_bl, tab_bc, tab_lz = st.tabs(
                ["Nearest Neighbor", "Bilinear", "Bicubic", "Lanczos4"]
            )
            with tab_nn:
                st.markdown("""
**Nearest Neighbor (Order 0)**

Nilai piksel diambil dari piksel terdekat saja tanpa interpolasi.

$$f(x,y) = f(\\lfloor x+0.5 \\rfloor, \\lfloor y+0.5 \\rfloor)$$

✅ Sangat cepat, mempertahankan nilai piksel asli.
❌ Artefak blok (*pixelation*) pada pembesaran, tepi bergerigi.
                """)
            with tab_bl:
                st.markdown("""
**Bilinear Interpolation (Order 1)**

Rata-rata berbobot dari 4 piksel terdekat di sekitar koordinat target.

$$f(x,y) = (1-a)(1-b)f_{00} + a(1-b)f_{10} + (1-a)bf_{01} + ab f_{11}$$

✅ Cepat, hasil halus, standar untuk mayoritas kasus.
❌ Dapat menyebabkan sedikit blur, tepi tidak setajam bicubic.
                """)
            with tab_bc:
                st.markdown("""
**Bicubic Interpolation (Order 3)**

Menggunakan 16 piksel tetangga (4×4) dengan polinomial kubik berbobot.

$$f(x,y) = \\sum_{i=-1}^{2}\\sum_{j=-1}^{2} a_{ij} \\cdot p(x-i) \\cdot p(y-j)$$

✅ Menghasilkan tepi yang lebih tajam, artefak minimum.
❌ Lebih lambat dari bilinear, sedikit *ringing* pada tepi kontras tinggi.
                """)
            with tab_lz:
                st.markdown("""
**Lanczos4 Interpolation**

Menggunakan 8×8 piksel tetangga dengan kernel sinc yang ditruncate.

$$L(x) = \\text{sinc}(x) \\cdot \\text{sinc}(x/a), \\quad |x| < a=4$$

✅ Kualitas terbaik, tajam, artefak aliasing minimal.
❌ Paling lambat secara komputasi dari semua metode.
                """)

    else:
        st.info("🔵 Atur parameter lalu klik **Jalankan Transformasi**.", icon="ℹ️")


# ─────────────────────────────────────────────────────────────────────────────
# MENU 3 — Analisis Kompresi & Kuantisasi (DCT)
# ─────────────────────────────────────────────────────────────────────────────

def render_menu_dct(img_rgb: np.ndarray):
    """Tab 3: DCT 8×8, kuantisasi, PSNR/MSE, analisis trade-off."""

    st.markdown('<div class="hero-header"><p class="hero-title">🗜️ Menu 3: <span>Kompresi &amp; Kuantisasi DCT</span></p><p class="hero-subtitle">JPEG-like Compression — DCT Blok 8×8 • PSNR • MSE • Trade-off Analysis</p></div>', unsafe_allow_html=True)

    # ── Panel Kontrol ────────────────────────────────────────────────
    with st.expander("⚙️ Parameter Kompresi", expanded=True):
        col_q, col_info = st.columns([2, 1])
        with col_q:
            quality_factor = st.slider(
                "Quality Factor (QF):",
                min_value=1, max_value=95, value=50, step=1,
                help="QF tinggi = kualitas tinggi, kompresi rendah. QF rendah = kualitas rendah, kompresi tinggi.",
            )
        with col_info:
            st.markdown(
                f"""<div class="metric-card" style="margin-top:0.5rem;">
                      <div class="metric-label">Quality Factor</div>
                      <div class="metric-value">{quality_factor}</div>
                      <div class="metric-unit">{"🟢 Tinggi" if quality_factor >= 50 else "🟡 Sedang" if quality_factor >= 25 else "🔴 Rendah"}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

        run_dct = st.button("🚀 Jalankan Kompresi DCT", type="primary", use_container_width=True)

    # ── Eksekusi ─────────────────────────────────────────────────────
    if run_dct or st.session_state.dct_result is not None:
        if run_dct:
            processor = CompressionAnalysisProcessor()
            with st.spinner("⚙️ Menghitung DCT blok 8×8 dan kuantisasi…"):
                try:
                    result: CompressionResult = processor.process(
                        img_rgb, quality_factor=quality_factor
                    )
                    st.session_state.dct_result = result
                except RuntimeError as e:
                    st.error(str(e))
                    return

        result: CompressionResult = st.session_state.dct_result
        if result is None:
            return

        st.markdown(
            f'<div style="margin-bottom:0.5rem;">{_exec_badge(result.elapsed_ms)} &nbsp; '
            f'<span class="chip">QF: {result.quality_factor}</span></div>',
            unsafe_allow_html=True,
        )

        # ── Metrik Evaluasi ─────────────────────────────────────────
        _section_title("📊 Metrik Evaluasi Objektif")
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            _metric_card("PSNR", f"{result.psnr_db:.2f}", "dB")
        with m2:
            _metric_card("MSE", f"{result.mse:.2f}", "piksel²")
        with m3:
            _metric_card("Quality Factor", str(result.quality_factor), "/95")
        with m4:
            _metric_card("Non-zero Koef.", f"{result.nonzero_ratio*100:.1f}", "%")
        with m5:
            _metric_card("Estimasi Rasio", f"{result.compression_ratio:.1f}", "×")

        # Gauge PSNR
        fig_gauge = _plot_psnr_gauge(result.psnr_db)
        st.pyplot(fig_gauge, use_container_width=True)
        plt.close(fig_gauge)

        st.markdown('<div class="pcd-divider"></div>', unsafe_allow_html=True)

        # ── Trade-off Analysis ───────────────────────────────────────
        _section_title("⚖️ Analisis Trade-off Kualitas vs Kompresi")
        st.markdown(
            f'<div class="tradeoff-box">{result.quality_assessment}</div>',
            unsafe_allow_html=True,
        )

        if result.psnr_db < 30:
            st.markdown(
                '<div class="warn-box">⚠️ <b>Peringatan:</b> PSNR di bawah 30 dB menunjukkan '
                'degradasi visual yang signifikan. Artefak blok DCT 8×8 akan terlihat jelas. '
                'Pertimbangkan untuk meningkatkan Quality Factor.</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="pcd-divider"></div>', unsafe_allow_html=True)

        # ── Perbandingan Citra ───────────────────────────────────────
        _section_title("Perbandingan Visual Citra Asli vs Terkompresi")
        col_o, col_c = st.columns(2)
        _display_image(col_o, result.original_rgb,    "📷 Citra Asli (Grayscale-rendered)")
        _display_image(col_c, result.compressed_rgb,  f"🗜️ Hasil Kompresi DCT (QF={result.quality_factor})")

        st.markdown('<div class="pcd-divider"></div>', unsafe_allow_html=True)

        # ── Visualisasi Koefisien DCT ────────────────────────────────
        _section_title("Visualisasi Koefisien DCT (Domain Frekuensi)")
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            fig_dct = _plot_dct_heatmap(
                cv2.cvtColor(result.dct_vis_rgb, cv2.COLOR_RGB2GRAY),
                "Koefisien DCT Asli (Log |DCT|)"
            )
            st.pyplot(fig_dct, use_container_width=True)
            st.markdown('<p class="img-caption">Energi terkonsentrasi di pojok kiri atas (DC + frekuensi rendah).</p>', unsafe_allow_html=True)
            plt.close(fig_dct)

        with col_d2:
            fig_qdct = _plot_dct_heatmap(
                cv2.cvtColor(result.quantized_dct_vis_rgb, cv2.COLOR_RGB2GRAY),
                "Koefisien DCT Setelah Kuantisasi"
            )
            st.pyplot(fig_qdct, use_container_width=True)
            st.markdown('<p class="img-caption">Banyak koefisien bernilai nol (hitam) = informasi yang dibuang.</p>', unsafe_allow_html=True)
            plt.close(fig_qdct)

        st.markdown('<div class="pcd-divider"></div>', unsafe_allow_html=True)

        # ── Penjelasan DCT ───────────────────────────────────────────
        _section_title("📖 Penjelasan Algoritma Kompresi DCT")
        with st.expander("Klik untuk membaca penjelasan langkah per langkah"):
            st.markdown(f"""
**Pipeline Kompresi JPEG-like (Quality Factor = {result.quality_factor})**

**Langkah 1 — Level Shift**
Nilai piksel dikurangi 128 agar terpusat di sekitar nol:
$$g(x,y) = f(x,y) - 128$$

**Langkah 2 — DCT 2D pada Blok 8×8**
$$D(u,v) = \\frac{{1}}{{4}} C(u) C(v) \\sum_{{x=0}}^{{7}} \\sum_{{y=0}}^{{7}} g(x,y) \\cos\\left(\\frac{{(2x+1)u\\pi}}{{16}}\\right) \\cos\\left(\\frac{{(2y+1)v\\pi}}{{16}}\\right)$$

Koefisien D(0,0) disebut **DC** (rata-rata blok), selebihnya disebut **AC**.

**Langkah 3 — Kuantisasi (Quality Factor = {result.quality_factor})**
$$D_q(u,v) = \\text{{round}}\\left(\\frac{{D(u,v)}}{{Q(u,v)}}\\right)$$

Matriks kuantisasi Q dibangun dengan skalar:
$$S = {"5000 / QF" if result.quality_factor < 50 else "200 - 2×QF"} = {5000//result.quality_factor if result.quality_factor < 50 else 200-2*result.quality_factor}$$

**Langkah 4 — De-kuantisasi & IDCT**
$$R(x,y) = \\text{{IDCT}}(D_q(u,v) \\times Q(u,v)) + 128$$

**Hasil:** {result.nonzero_ratio*100:.1f}% koefisien non-zero → estimasi kompresi **{result.compression_ratio:.1f}×**.
""")

    else:
        st.info("🔵 Atur Quality Factor lalu klik **Jalankan Kompresi DCT**.", icon="ℹ️")


# ─────────────────────────────────────────────────────────────────────────────
# Main — Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Entry point utama aplikasi Streamlit PCD Dashboard."""
    SessionStateManager.init()
    render_sidebar()

    # ── Header Utama ─────────────────────────────────────────────────
    st.markdown(
        """<div class="hero-header">
             <p class="hero-title">🔬 Repositori Tugas Akhir — <span>Pengolahan Citra Digital</span></p>
             <p class="hero-subtitle">
               Aplikasi web interaktif untuk eksplorasi domain frekuensi, transformasi geometri,
               dan analisis kompresi citra secara visual dan kuantitatif.
             </p>
           </div>""",
        unsafe_allow_html=True,
    )

    # ── Cek Citra ────────────────────────────────────────────────────
    loaded, img_rgb = SessionStateManager.get_image()
    if not loaded or img_rgb is None:
        st.warning("👈 Silakan muat citra terlebih dahulu melalui **sidebar** di sebelah kiri.", icon="⚠️")
        return

    # ── Tab Navigasi ─────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📡  Menu 1 — Domain Frekuensi (FFT)",
        "📐  Menu 2 — Transformasi Geometri",
        "🗜️  Menu 3 — Kompresi DCT",
    ])

    with tab1:
        render_menu_fft(img_rgb)

    with tab2:
        render_menu_geo(img_rgb)

    with tab3:
        render_menu_dct(img_rgb)

    # ── Footer ───────────────────────────────────────────────────────
    st.markdown(
        """<div style="text-align:center; padding: 2rem 0 0.5rem 0;
                       font-size:0.75rem; color:#505070;">
             PCD Dashboard · Mata Kuliah Pengolahan Citra Digital ·
             Built with Streamlit + OpenCV + NumPy + SciPy
           </div>""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
