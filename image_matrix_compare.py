import numpy as np
import time
from PIL import Image
import sys
import os


# ──────────────────────────────────────────────
#  ALGORITHM IMPLEMENTATIONS
# ──────────────────────────────────────────────

def brute_force_matmul(A, B):
    n, m, p = len(A), len(B[0]), len(B)
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                C[i][j] += A[i][k] * B[k][j]
    return C


def _strassen(A, B):
    n = A.shape[0]
    if n <= 64:
        return A @ B
    if n % 2 != 0:
        A = np.pad(A, ((0, 1), (0, 1)))
        B = np.pad(B, ((0, 1), (0, 1)))
        orig, n = n, n + 1
    else:
        orig = n
    mid = n // 2
    a11, a12 = A[:mid, :mid], A[:mid, mid:]
    a21, a22 = A[mid:, :mid], A[mid:, mid:]
    b11, b12 = B[:mid, :mid], B[:mid, mid:]
    b21, b22 = B[mid:, :mid], B[mid:, mid:]
    m1 = _strassen(a11 + a22, b11 + b22)
    m2 = _strassen(a21 + a22, b11)
    m3 = _strassen(a11,        b12 - b22)
    
    m4 = _strassen(a22,        b21 - b11)
    m5 = _strassen(a11 + a12,  b22)
    m6 = _strassen(a21 - a11,  b11 + b12)
    m7 = _strassen(a12 - a22,  b21 + b22)
    C = np.empty((n, n))
    C[:mid, :mid] = m1 + m4 - m5 + m7
    C[:mid, mid:] = m3 + m5
    C[mid:, :mid] = m2 + m4
    C[mid:, mid:] = m1 - m2 + m3 + m6
    return C[:orig, :orig]


def strassen_matmul(A, B):
    return _strassen(np.array(A, dtype=np.float64),
                     np.array(B, dtype=np.float64))


def numpy_matmul(A, B):
    return np.array(A, dtype=np.float64) @ np.array(B, dtype=np.float64)


# ──────────────────────────────────────────────
#  COLOR TRANSFORM MATRICES
# ──────────────────────────────────────────────

TRANSFORMS = {
    "1": ("Sepia Tone", np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])),
    "2": ("Cool Tone", np.array([
        [0.8, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 1.20]
    ])),
    "3": ("Warm Tone", np.array([
        [1.20, 0.05, 0.00],
        [0.00, 1.00, 0.00],
        [0.00, 0.00, 0.80]
    ])),
    "4": ("Grayscale", np.array([
        [0.299, 0.587, 0.114],
        [0.299, 0.587, 0.114],
        [0.299, 0.587, 0.114]
    ])),
    "5": ("Invert-Tint", np.array([
        [ 0.50, -0.25, -0.25],
        [-0.25,  0.50, -0.25],
        [-0.25, -0.25,  0.50]
    ])),
}


# ──────────────────────────────────────────────
#  CORE PROCESSING
# ──────────────────────────────────────────────

def load_image_8bit(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def apply_transform(image, M, method):
    h, w, _ = image.shape
    pixels = image.reshape(-1, 3).astype(np.float64) / 255.0
    Mt = M.T

    if method == "brute_force":
        chunk = min(len(pixels), 256)
        out_chunks = []
        for i in range(0, len(pixels), chunk):
            block = pixels[i:i+chunk].tolist()
            result = brute_force_matmul(block, Mt.tolist())
            out_chunks.append(np.array(result))
        result = np.vstack(out_chunks)
    elif method == "strassen":
        result = strassen_matmul(pixels.tolist(), Mt.tolist())
    else:
        result = numpy_matmul(pixels.tolist(), Mt.tolist())

    result = np.clip(result, 0, 1)
    return (result.reshape(h, w, 3) * 255).astype(np.uint8)


def benchmark_matmul(size, repeat=3):
    np.random.seed(42)
    A = np.random.randint(0, 256, (size, size)).astype(float).tolist()
    B = np.random.randint(0, 256, (size, size)).astype(float).tolist()
    results = {}
    for name, fn in [("brute_force", brute_force_matmul),
                     ("strassen",    strassen_matmul),
                     ("numpy",       numpy_matmul)]:
        if name == "brute_force" and size > 128:
            results[name] = None
            continue
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn(A, B)
            times.append(time.perf_counter() - t0)
        results[name] = np.mean(times) * 1000
    return results


def save_result_image(array, base_path, suffix):
    out_path = os.path.splitext(base_path)[0] + f"_{suffix}.png"
    Image.fromarray(array).save(out_path)
    return out_path


# ──────────────────────────────────────────────
#  CLI INTERFACE
# ──────────────────────────────────────────────

def menu_transform():
    print("\nPilih transformasi warna:")
    for k, (name, _) in TRANSFORMS.items():
        print(f"  [{k}] {name}")
    choice = input("Masukkan nomor [1-5]: ").strip()
    return TRANSFORMS.get(choice, TRANSFORMS["1"])


def run():
    print("=" * 55)
    print("  IMAGE MATRIX TRANSFORM — Brute Force vs Strassen vs NumPy")
    print("=" * 55)

    # Input citra
    path = input("\nMasukkan path citra (contoh: foto.jpg): ").strip().strip('"')
    if not os.path.exists(path):
        print(f"[ERROR] File tidak ditemukan: {path}")
        sys.exit(1)

    image = load_image_8bit(path)
    print(f"[OK] Citra dimuat: {image.shape[1]}x{image.shape[0]} px, 8-bit RGB")

    transform_name, M = menu_transform()
    print(f"\nTransformasi: {transform_name}")

    # Jalankan 3 algoritma
    print("\n  Algoritma       | Waktu (ms)  | Output")
    print("  " + "-" * 50)

    outputs = {}
    for method in ["brute_force", "strassen", "numpy"]:
        t0 = time.perf_counter()
        result = apply_transform(image, M, method)
        elapsed = (time.perf_counter() - t0) * 1000
        out_path = save_result_image(result, path, method)
        outputs[method] = (elapsed, out_path)
        label = method.replace("_", " ").title()
        print(f"  {label:<16} | {elapsed:>8.2f} ms  | {os.path.basename(out_path)}")

    # Benchmark perkalian matriks murni
    print("\n  Benchmark matriks murni (64x64):")
    bench = benchmark_matmul(64, repeat=3)
    for name, ms in bench.items():
        val = f"{ms:.4f} ms" if ms is not None else "skipped"
        print(f"    {name:<14}: {val}")

    bf_ms = bench.get("brute_force")
    np_ms = bench.get("numpy")
    if bf_ms and np_ms:
        print(f"\n  NumPy {bf_ms/np_ms:.1f}x lebih cepat dari Brute Force (n=64)")

    print("\n[SELESAI] Hasil tersimpan di folder yang sama dengan citra input.")
    print("=" * 55)


if __name__ == "__main__":
    run()
