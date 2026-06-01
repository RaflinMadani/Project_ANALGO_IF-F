import random
import math

# ── 1. Matriks Random ────────────────────────────────────────────────────────
def random_matrix(n: int, m: int,
                  lo: float = -10.0, hi: float = 10.0,
                  seed: int = None) -> list:
    rng = random.Random(seed)
    return [
        [rng.uniform(lo, hi) for _ in range(m)]
        for _ in range(n)
    ]

# ── 2. Matriks DCT-II ────────────────────────────────────────────────────────
def dct_matrix(N: int = 8) -> list:
    D = []
    for k in range(N):
        row = []
        for n in range(N):
            if k == 0:
                val = math.sqrt(1.0 / N)
            else:
                val = math.sqrt(2.0 / N) * math.cos(math.pi * k * (2*n + 1) / (2*N))
            row.append(val)
        D.append(row)
    return D

def dct_matrix_transpose(N: int = 8) -> list:
    D = dct_matrix(N)
    return [[D[j][i] for j in range(N)] for i in range(N)]

def random_block(N: int = 8, seed: int = None) -> list:
    rng = random.Random(seed)
    return [
        [rng.uniform(-128.0, 127.0) for _ in range(N)]
        for _ in range(N)
    ]

# ── 3. Utilitas tampilan ─────────────────────────────────────────────────────
def matrix_from_user_input(rows_str: str) -> list:
    lines = [ln.strip() for ln in rows_str.strip().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Input kosong.")

    matrix = []
    for ln in lines:
        # support spasi atau koma sebagai separator
        parts = ln.replace(',', ' ').split()
        try:
            row = [float(x) for x in parts]
        except ValueError:
            raise ValueError(f"Nilai tidak valid di baris: '{ln}'")
        matrix.append(row)

    # Cek semua baris sama panjang
    widths = [len(r) for r in matrix]
    if len(set(widths)) > 1:
        raise ValueError(
            f"Panjang baris tidak konsisten: {widths}"
        )

    return matrix

def format_matrix(M: list, precision: int = 4) -> str:
    if not M:
        return "(kosong)"
    rows = []
    for row in M:
        cells = [f"{v:>{precision+6}.{precision}f}" for v in row]
        rows.append("  [ " + "  ".join(cells) + " ]")
    return "\n".join(rows)