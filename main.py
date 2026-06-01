import streamlit as st
import time
import math

import matriks_brute_force           as bf
import matriks_brute_force_optimized as bfo
import matriks_strassen              as st_algo
import matriks_numpy                 as npy
import generate_matriks              as gen

# ─────────────────────────────────────────────────────────────────────────────
#  KONFIGURASI HALAMAN
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Perbandingan Algoritma Perkalian Matriks",
    page_icon="🧮",
    layout="wide",
)

st.title("🧮 Perbandingan Algoritma Perkalian Matriks")
st.caption(
    "Brute Force  ·  Brute Force Optimized  ·  Strassen  ·  NumPy (BLAS)"
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR — KONFIGURASI INPUT
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Konfigurasi")

    mode = st.radio(
        "Sumber matriks",
        ["Input manual", "Random", "DCT 2D (8×8)"],
        index=1,
    )

    if mode == "Random":
        n_size = st.slider("Ukuran matriks (n×n)", min_value=2, max_value=64, value=4, step=1)
        seed   = st.number_input("Seed random", value=42, step=1)
        lo     = st.number_input("Nilai minimum", value=-10.0, step=1.0)
        hi     = st.number_input("Nilai maksimum", value=10.0, step=1.0)

    elif mode == "DCT 2D (8×8)":
        st.info(
            "Menggunakan matriks basis DCT-II ortogonal 8×8.\n\n"
            "Ini adalah matriks yang dipakai standar JPEG untuk kompresi gambar:\n"
            "`C = D × blok × D^T`"
        )

    st.divider()

    run_bf  = st.checkbox("Brute Force  O(n³)",           value=True)
    run_bfo = st.checkbox("Brute Force Optimized  O(n³)*", value=True)
    run_st  = st.checkbox("Strassen  O(n^2.807)",         value=True)
    run_np  = st.checkbox("NumPy BLAS  O(n^2.37)",        value=True)

    repeat = st.slider("Pengulangan pengukuran", min_value=1, max_value=5, value=1)

    st.divider()
    jalankan = st.button("▶ Jalankan", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
#  PANEL INPUT MATRIKS
# ─────────────────────────────────────────────────────────────────────────────
A, B = None, None
input_error = None

if mode == "Input manual":
    st.subheader("Input Matriks")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Matriks A**")
        st.caption("Pisahkan nilai dengan spasi, baris baru untuk baris berikutnya")
        raw_A = st.text_area(
            "Matriks A",
            value="1 2 3\n4 5 6\n7 8 9",
            height=120,
            label_visibility="collapsed",
            key="input_A",
        )

    with col2:
        st.markdown("**Matriks B**")
        st.caption("Jumlah kolom A harus sama dengan jumlah baris B")
        raw_B = st.text_area(
            "Matriks B",
            value="9 8 7\n6 5 4\n3 2 1",
            height=120,
            label_visibility="collapsed",
            key="input_B",
        )

    try:
        A = gen.matrix_from_user_input(raw_A)
        B = gen.matrix_from_user_input(raw_B)

        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"Dimensi A: {len(A)}×{len(A[0])}")
        with col2:
            st.caption(f"Dimensi B: {len(B)}×{len(B[0])}")

        # Validasi dimensi
        if len(A[0]) != len(B):
            input_error = (
                f"Kolom A ({len(A[0])}) harus sama dengan baris B ({len(B)})"
            )
    except ValueError as e:
        input_error = str(e)

    if input_error:
        st.error(f"❌ {input_error}")

elif mode == "Random":
    A = gen.random_matrix(n_size, n_size, lo, hi, seed=int(seed))
    B = gen.random_matrix(n_size, n_size, lo, hi, seed=int(seed) + 1)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Matriks A")
        st.caption(f"Dimensi: {len(A)}×{len(A[0])}")
        if n_size <= 8:
            st.dataframe(A, use_container_width=True, hide_index=False)

    with col2:
        st.subheader("Matriks B")
        st.caption(f"Dimensi: {len(B)}×{len(B[0])}")
        if n_size <= 8:
            st.dataframe(B, use_container_width=True, hide_index=False)

    if n_size > 8:
        st.info(f"Matriks {n_size}×{n_size} terlalu besar untuk ditampilkan.")

else:  # DCT 2D
    A = gen.dct_matrix(8)
    B = gen.random_block(8, seed=42)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Matriks A — Basis DCT (8×8)")
        st.caption("D[k][n] = sqrt(2/N)·cos(π·k·(2n+1)/2N)")
        st.dataframe(
            [[f"{v:.4f}" for v in row] for row in A],
            use_container_width=True,
            hide_index=False,
        )
    with col2:
        st.subheader("Matriks B — Blok Piksel (8×8)")
        st.caption("Nilai piksel ter-level-shift: rentang [-128, 127]")
        st.dataframe(
            [[f"{v:.2f}" for v in row] for row in B],
            use_container_width=True,
            hide_index=False,
        )

# ─────────────────────────────────────────────────────────────────────────────
#  EKSEKUSI ALGORITMA
# ─────────────────────────────────────────────────────────────────────────────
def run_algorithm(fn, A, B, repeat):
    """Jalankan fungsi multiply sebanyak `repeat` kali, kembalikan (hasil, rata_ms)."""
    result = None
    total  = 0.0
    for _ in range(repeat):
        t0     = time.perf_counter()
        result = fn(A, B)
        total += time.perf_counter() - t0
    return result, (total / repeat) * 1000   # rata-rata dalam ms

def max_diff(C1, C2):
    """Hitung selisih maksimum antara dua matriks (untuk verifikasi kebenaran)."""
    max_d = 0.0
    for i in range(len(C1)):
        for j in range(len(C1[0])):
            d = abs(C1[i][j] - C2[i][j])
            if d > max_d:
                max_d = d
    return max_d

if jalankan and A is not None and input_error is None:
    st.divider()
    st.subheader("📊 Hasil Perbandingan")

    ALGORITHMS = []
    if run_bf:
        ALGORITHMS.append(("Brute Force",           "O(n³)",       bf.multiply,  "#E24B4A"))
    if run_bfo:
        ALGORITHMS.append(("BF Optimized",          "O(n³)*",      bfo.multiply, "#EF9F27"))
    if run_st:
        ALGORITHMS.append(("Strassen",              "O(n^2.807)",  st_algo.multiply, "#7F77DD"))
    if run_np:
        ALGORITHMS.append(("NumPy BLAS",            "O(n^2.37)",   npy.multiply, "#1D9E75"))

    if not ALGORITHMS:
        st.warning("Pilih minimal satu algoritma.")
        st.stop()

    results  = {}
    times_ms = {}

    progress = st.progress(0, text="Menjalankan algoritma...")
    for idx, (name, _, fn, _) in enumerate(ALGORITHMS):
        try:
            C, ms = run_algorithm(fn, A, B, repeat)
            results[name]  = C
            times_ms[name] = ms
        except Exception as e:
            st.error(f"Error pada {name}: {e}")
            results[name]  = None
            times_ms[name] = None
        progress.progress((idx + 1) / len(ALGORITHMS), text=f"Selesai: {name}")

    progress.empty()

    # ── Metric cards waktu ────────────────────────────────────────────────
    cols = st.columns(len(ALGORITHMS))
    for col, (name, complexity, _, color) in zip(cols, ALGORITHMS):
        ms = times_ms.get(name)
        with col:
            if ms is not None:
                st.metric(
                    label=f"{name}  `{complexity}`",
                    value=f"{ms:.3f} ms",
                )
            else:
                st.metric(label=name, value="Error")

    # ── Grafik batang waktu ───────────────────────────────────────────────
    valid = {n: t for n, t in times_ms.items() if t is not None}
    if len(valid) > 1:
        st.divider()
        st.subheader("⏱ Perbandingan Waktu")

        import importlib.util
        has_altair = importlib.util.find_spec("altair") is not None

        if has_altair:
            import altair as alt
            import pandas as pd
            df = pd.DataFrame({
                "Algoritma": list(valid.keys()),
                "Waktu (ms)": list(valid.values()),
            })
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("Algoritma:N", sort=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Waktu (ms):Q"),
                    color=alt.Color(
                        "Algoritma:N",
                        scale=alt.Scale(
                            domain=list(valid.keys()),
                            range=[c for _, _, _, c in ALGORITHMS if times_ms.get(_) is not None],
                        ),
                        legend=None,
                    ),
                    tooltip=["Algoritma", alt.Tooltip("Waktu (ms):Q", format=".4f")],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            # Fallback: bar chart teks
            max_t = max(valid.values())
            for name, ms in valid.items():
                bar_len = int(ms / max_t * 40)
                st.text(f"{name:<22} {'█' * bar_len} {ms:.3f} ms")

    # ── Speedup relatif terhadap Brute Force ─────────────────────────────
    if "Brute Force" in valid and len(valid) > 1:
        st.divider()
        st.subheader("🚀 Speedup vs Brute Force")
        bf_time = valid["Brute Force"]
        cols    = st.columns(len(valid) - 1)
        i       = 0
        for name, ms in valid.items():
            if name == "Brute Force":
                continue
            with cols[i]:
                speedup = bf_time / ms if ms > 0 else float("inf")
                st.metric(
                    label=name,
                    value=f"{speedup:.2f}×",
                    delta=f"{ms:.3f} ms",
                    delta_color="inverse",
                )
            i += 1

    # ── Verifikasi kebenaran ──────────────────────────────────────────────
    st.divider()
    st.subheader("✅ Verifikasi Kebenaran")

    ref_name   = next(iter(results))
    ref_result = results[ref_name]

    all_correct = True
    vcols = st.columns(len(results))
    for col, (name, result) in zip(vcols, results.items()):
        with col:
            if result is None:
                st.error(f"**{name}**\nError")
                all_correct = False
            elif name == ref_name:
                st.success(f"**{name}**\nReferensi")
            else:
                diff = max_diff(ref_result, result)
                if diff < 1e-6:
                    st.success(f"**{name}**\nSelisih: {diff:.2e} ✓")
                else:
                    st.error(f"**{name}**\nSelisih: {diff:.2e} ✗")
                    all_correct = False

    if all_correct:
        st.success("Semua algoritma menghasilkan matriks yang identik (error < 1e-6).")

    # ── Tampilkan hasil matriks C ─────────────────────────────────────────
    show_result = st.checkbox("Tampilkan matriks hasil C", value=False)
    if show_result:
        st.divider()
        st.subheader("Matriks Hasil C = A × B")
        ref_C = next(r for r in results.values() if r is not None)
        n_res = len(ref_C)
        if n_res <= 12:
            st.dataframe(
                [[f"{v:.4f}" for v in row] for row in ref_C],
                use_container_width=True,
                hide_index=False,
            )
        else:
            st.info(
                f"Matriks hasil {n_res}×{len(ref_C[0])} terlalu besar untuk ditampilkan."
                " Centang hanya untuk matriks kecil."
            )

    # ── Kompleksitas Waktu ────────────────────────────────────────────────
    st.divider()
    with st.expander("📖 Penjelasan Kompleksitas Waktu"):
        st.markdown("""
| Algoritma | Kompleksitas | Keterangan |
|---|---|---|
| **Brute Force** | O(n³) | Triple loop i→j→k. Akses B column-wise → banyak cache miss |
| **BF Optimized** | O(n³)* | Loop reorder i→k→j + prefetch A[i][k]. Akses B row-wise → cache-friendly |
| **Strassen** | O(n^2.807) | Divide & conquer. 7 perkalian rekursif (bukan 8). Keuntungan nyata mulai n ≈ 64+ |
| **NumPy BLAS** | O(n^2.37) | BLAS dgemm. Cache-blocked + SIMD/AVX + multi-thread. Tidak ada loop Python |
        """)

elif jalankan and input_error:
    st.error(f"Tidak dapat menjalankan: {input_error}")

elif not jalankan:
    st.info("← Atur konfigurasi di sidebar, lalu tekan **▶ Jalankan**.")
