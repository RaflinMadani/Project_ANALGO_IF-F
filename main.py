import streamlit as st
import time
import numpy as np
import pandas as pd
import altair as alt
import cv2

import matriks_brute_force           as bf
import matriks_brute_force_optimized as bfo
import matriks_strassen              as st_algo
import matriks_winograd              as wg_algo
import matriks_numpy                 as npy
import generate_matriks              as gen

st.set_page_config(page_title="Perbandingan Algoritma Perkalian Matriks", page_icon="🧮", layout="wide")
st.title("🧮 Perbandingan Algoritma Perkalian Matriks")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    mode = st.radio("Sumber Matriks", ["Input Manual", "Random", "Transformasi Warna (RGB → Grayscale)"], index=2)

    if mode == "Input Manual":
        col1, col2 = st.columns(2)
        with col1:
            rows_a = st.number_input("Baris A", 1, 8, 3, 1)
            cols_a = st.number_input("Kolom A", 1, 8, 3, 1)
        with col2:
            rows_b = st.number_input("Baris B", 1, 8, 3, 1)
            cols_b = st.number_input("Kolom B", 1, 8, 3, 1)
        if cols_a != rows_b:
            st.error(f"Kolom A ({cols_a}) ≠ Baris B ({rows_b})")
        st.subheader("Matriks A")
        df_a = pd.DataFrame([[1.0]*cols_a for _ in range(rows_a)], columns=[f"c{j+1}" for j in range(cols_a)])
        edited_a = st.data_editor(df_a, key="edit_A", hide_index=True)
        st.subheader("Matriks B")
        df_b = pd.DataFrame([[1.0]*cols_b for _ in range(rows_b)], columns=[f"c{j+1}" for j in range(cols_b)])
        edited_b = st.data_editor(df_b, key="edit_B", hide_index=True)

    elif mode == "Random":
        n_size = st.number_input("Ukuran (n×n)", 2, 8192, 4, 1)
        st.caption("Nilai acak positif 1–10")

    else:  # RGB → Grayscale
        uploaded_file = st.file_uploader("Upload Citra Berwarna", ["jpg","png","jpeg","bmp"])
        if uploaded_file:
            st.success("✅ Citra terupload")
        else:
            st.info("Upload citra berwarna")

    st.divider()
    run_bf  = st.checkbox("Brute Force", True)
    run_bfo = st.checkbox("Brute Force Optimized", True)
    run_st  = st.checkbox("Strassen", True)
    run_wg  = st.checkbox("Winograd", True)
    run_np  = st.checkbox("NumPy BLAS", True)

    st.divider()
    jalankan = st.button("▶ Jalankan", type="primary", use_container_width=True)

# ── MAIN LOGIC ──────────────────────────────────────────────────────────────
if jalankan:
    A, B = None, None
    img_original = None
    transform_mode = False

    try:
        if mode == "Input Manual":
            A = edited_a.values.tolist()
            B = edited_b.values.tolist()
            if len(A[0]) != len(B):
                st.error("Dimensi tidak cocok!")
                st.stop()

        elif mode == "Random":
            st.toast(f"🔄 Membuat matriks {n_size}×{n_size}...", icon="⚡")
            A = gen.random_matrix(n_size, n_size, lo=1, hi=10, seed=42)
            B = gen.random_matrix(n_size, n_size, lo=1, hi=10, seed=7)

        elif mode == "Transformasi Warna (RGB → Grayscale)":
            if uploaded_file is None:
                st.error("Upload citra dulu!")
                st.stop()
            st.toast("🖼️ Memproses citra...", icon="📸")
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                st.error("Gagal baca citra")
                st.stop()
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]
            b, g, r = cv2.split(img)
            r_flat = r.reshape(-1).astype(np.float64)
            g_flat = g.reshape(-1).astype(np.float64)
            b_flat = b.reshape(-1).astype(np.float64)
            img_matrix = np.array([r_flat, g_flat, b_flat])  # 3×N
            A = [[0.299, 0.587, 0.114]]  # 1×3 
            B = img_matrix.tolist()
            transform_mode = True
            st.info(f"Citra {W}×{H}, total piksel: {H*W}")

    except Exception as e:
        st.error(f"❌ Gagal menyiapkan data: {e}")
        st.stop()

    # ── Daftar algoritma ────────────────────────────────────────────────────
    algorithms = []
    if run_bf:  algorithms.append(("Brute Force", bf.multiply, "#E24B4A"))
    if run_bfo: algorithms.append(("BF Optimized", bfo.multiply, "#EF9F27"))
    if run_st:  algorithms.append(("Strassen", st_algo.multiply, "#7F77DD"))
    if run_wg:  algorithms.append(("Winograd", wg_algo.multiply, "#FFA500"))
    if run_np:  algorithms.append(("NumPy BLAS", npy.multiply, "#1D9E75"))

    if not algorithms:
        st.warning("Pilih minimal satu algoritma")
        st.stop()

    # ── Eksekusi & ukur waktu ──────────────────────────────────────────────
    times, results, errors = {}, {}, {}
    progress = st.progress(0, text="Menjalankan algoritma...")

    for idx, (name, func, _) in enumerate(algorithms):
        try:
            start = time.perf_counter()
            C = func(A, B)  # Semua algoritma dijalankan pada data yang sama
            elapsed = time.perf_counter() - start
            times[name] = elapsed
            results[name] = C
            errors[name] = None
        except Exception as e:
            times[name] = None
            results[name] = None
            errors[name] = str(e)
        progress.progress((idx+1)/len(algorithms), text=f"Selesai: {name}")
    progress.empty()

    # ── Tampilkan hasil ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Hasil Perbandingan")

    cols = st.columns(len(algorithms))
    for col, (name, _, _) in zip(cols, algorithms):
        t = times.get(name)
        with col:
            if t is not None:
                st.metric(name, f"{t:.4f} detik", f"{t*1000:.2f} ms")
            else:
                st.metric(name, f"❌ Error: {errors.get(name, '')}")

    valid_times = {n: t for n, t in times.items() if t is not None}
    if len(valid_times) > 1:
        st.divider()
        st.subheader("⏱ Perbandingan Waktu")
        df = pd.DataFrame({
            "Algoritma": list(valid_times.keys()),
            "Waktu (detik)": list(valid_times.values())
        }).sort_values("Waktu (detik)", ascending=False)
        color_scale = alt.Scale(
            domain=[name for name, _, _ in algorithms if times.get(name)],
            range=[color for _, _, color in algorithms if times.get(name)]
        )
        st.altair_chart(
            alt.Chart(df).mark_bar().encode(
                x=alt.X("Waktu (detik):Q", title="Waktu (detik)"),
                y=alt.Y("Algoritma:N", sort=None, title=None),
                color=alt.Color("Algoritma:N", scale=color_scale, legend=None),
                tooltip=["Algoritma", alt.Tooltip("Waktu (detik):Q", format=".6f")]
            ).properties(height=300),
            use_container_width=True
        )

    # ── Output spesifik per mode ──────────────────────────────────────────
    st.divider()
    if mode == "Input Manual":
        ref = next((n for n, t in times.items() if t is not None), None)
        if ref and results[ref] is not None:
            st.subheader("Matriks Hasil (C = A × B)")
            if len(results[ref]) <= 8:
                st.dataframe(pd.DataFrame(results[ref]))
            else:
                st.info("Matriks terlalu besar untuk ditampilkan")

    elif mode == "Transformasi Warna (RGB → Grayscale)":
        st.subheader("Citra Asli vs Hasil Grayscale")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Citra Asli (RGB)")
            st.image(img_original, use_container_width=True)
        with col2:
            st.caption("Hasil Grayscale")
            ref = next((n for n, t in times.items() if t is not None), None)
            if ref and results[ref] is not None:
                gray_vals = np.array(results[ref]).reshape(-1)
                H, W = img_original.shape[:2]
                gray_img = gray_vals.reshape(H, W).astype(np.uint8)
                st.image(gray_img, use_container_width=True)
            else:
                st.warning("Tidak ada hasil dari algoritma manapun")

    st.success("✅ Selesai!")

else:
    st.info("👈 Atur konfigurasi di sidebar, lalu tekan ▶ Jalankan")