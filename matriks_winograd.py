import concurrent.futures
from typing import List, Tuple

# ── helper aritmetika matriks (list-based, tanpa library) ──────────────────
def _add(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    n, m = len(A), len(A[0])
    return [[A[i][j] + B[i][j] for j in range(m)] for i in range(n)]

def _sub(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    n, m = len(A), len(A[0])
    return [[A[i][j] - B[i][j] for j in range(m)] for i in range(n)]

def _split(M: List[List[float]]) -> Tuple:
    mid = len(M) // 2
    M11 = [row[:mid] for row in M[:mid]]
    M12 = [row[mid:] for row in M[:mid]]
    M21 = [row[:mid] for row in M[mid:]]
    M22 = [row[mid:] for row in M[mid:]]
    return M11, M12, M21, M22

def _join(C11: List[List[float]], C12: List[List[float]], 
          C21: List[List[float]], C22: List[List[float]]) -> List[List[float]]:
    top = [r1 + r2 for r1, r2 in zip(C11, C12)]
    bottom = [r1 + r2 for r1, r2 in zip(C21, C22)]
    return top + bottom

def _base_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Brute Force Optimized (i-k-j + pre-fetch) sebagai base case"""
    n, p = len(A), len(A[0])
    m = len(B[0])
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for k in range(p):
            a_ik = A[i][k]
            for j in range(m):
                C[i][j] += a_ik * B[k][j]
    return C

# ── konfigurasi ──────────────────────────────────────────────────────────────
_BASE_SIZE = 64         
_MT_THRESHOLD = 64       
_MAX_WORKERS = 12          

def _winograd_rec(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    n = len(A)

    # Base case
    if n <= _BASE_SIZE:
        return _base_multiply(A, B)

    # Divide
    A11, A12, A21, A22 = _split(A)
    B11, B12, B21, B22 = _split(B)

    # Winograd pre‑computations (mengurangi jumlah penjumlahan)
    S1 = _add(A11, A22)          # A11 + A22
    S2 = _add(B11, B22)          # B11 + B22
    S3 = _sub(A21, A11)          # A21 - A11
    S4 = _sub(B12, B22)          # B12 - B22
    S5 = _add(A11, A12)          # A11 + A12
    S6 = _sub(B21, B22)          # B21 - B22
    S7 = _sub(A12, A22)          # A12 - A22
    S8 = _sub(A21, A22)          # A21 - A22
    S9 = _add(B11, B12)          # B11 + B12
    S10 = _add(B21, B22)         # B21 + B22

    # 7 perkalian rekursif (M1 – M7)
    tasks = [
        (S1, S2),                # M1 = (A11+A22)(B11+B22)
        (_add(A21, A22), B11),   # M2 = (A21+A22)·B11
        (A11, S4),               # M3 = A11·(B12-B22)
        (A22, S6),               # M4 = A22·(B21-B22)
        (S5, B22),               # M5 = (A11+A12)·B22
        (S3, S9),                # M6 = (A21-A11)·(B11+B12)
        (S7, S10)                # M7 = (A12-A22)·(B21+B22)
    ]

    # Eksekusi paralel jika ukuran cukup besar
    if n >= _MT_THRESHOLD:
        with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
            futures = [ex.submit(_winograd_rec, a, b) for a, b in tasks]
            M1, M2, M3, M4, M5, M6, M7 = [f.result() for f in futures]
    else:
        M1, M2, M3, M4, M5, M6, M7 = (_winograd_rec(a, b) for a, b in tasks)

    # Combine (Winograd – 15 penjumlahan/pengurangan di level ini)
    C11 = _sub(_add(_add(M1, M4), M7), M5)
    C12 = _add(M3, M5)
    C21 = _add(M2, M4)
    C22 = _add(_sub(_add(M1, M3), M2), M6)

    return _join(C11, C12, C21, C22)

# ── entry point ──────────────────────────────────────────────────────────────
def multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    n_A, p_A = len(A), len(A[0])
    p_B, m_B = len(B), len(B[0])

    if p_A != p_B:
        raise ValueError(f"Dimensi tidak cocok: A({n_A}x{p_A}) ≠ B({p_B}x{m_B})")

    # Pad ke power-of-2
    max_dim = max(n_A, p_A, m_B)
    size = 1
    while size < max_dim:
        size *= 2

    A_pad = [[0.0] * size for _ in range(size)]
    B_pad = [[0.0] * size for _ in range(size)]

    for i in range(n_A):
        for j in range(p_A):
            A_pad[i][j] = float(A[i][j])
    for i in range(p_B):
        for j in range(m_B):
            B_pad[i][j] = float(B[i][j])

    C_pad = _winograd_rec(A_pad, B_pad)

    return [C_pad[i][:m_B] for i in range(n_A)]