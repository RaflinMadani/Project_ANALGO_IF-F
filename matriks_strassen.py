# ── helper aritmetika matriks (list-based, tanpa library) ────────────────────
def _add(A: list, B: list) -> list:
    """Penjumlahan dua matriks elemen per elemen."""
    n, m = len(A), len(A[0])
    return [[A[i][j] + B[i][j] for j in range(m)] for i in range(n)]

def _sub(A: list, B: list) -> list:
    """Pengurangan dua matriks elemen per elemen."""
    n, m = len(A), len(A[0])
    return [[A[i][j] - B[i][j] for j in range(m)] for i in range(n)]

def _split(M: list) -> tuple:
    """
    Bagi matriks n×n menjadi 4 sub-matriks n/2 × n/2.
    Mengembalikan (M11, M12, M21, M22).
    """
    mid = len(M) // 2
    M11 = [row[:mid] for row in M[:mid]]
    M12 = [row[mid:] for row in M[:mid]]
    M21 = [row[:mid] for row in M[mid:]]
    M22 = [row[mid:] for row in M[mid:]]
    return M11, M12, M21, M22

def _join(C11: list, C12: list, C21: list, C22: list) -> list:
    top    = [r1 + r2 for r1, r2 in zip(C11, C12)]
    bottom = [r1 + r2 for r1, r2 in zip(C21, C22)]
    return top + bottom

def _base_multiply(A: list, B: list) -> list:
    n = len(A)
    p = len(A[0])
    m = len(B[0])
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for k in range(p):
            a_ik = A[i][k]
            for j in range(m):
                C[i][j] += a_ik * B[k][j]
    return C

# ── rekursi inti Strassen ─────────────────────────────────────────────────────
_BASE_SIZE = 64   # ukuran minimum sebelum turun ke base case skalar

def _strassen_rec(A: list, B: list) -> list:
    n = len(A)
    
    # Batasi ukuran maksimum untuk menghindari crash
    MAX_SIZE = 8192
    if n > MAX_SIZE:
        raise ValueError(f"Ukuran matriks terlalu besar untuk Strassen: {n}×{n} > {MAX_SIZE}×{MAX_SIZE}")

    # Base case: gunakan perkalian skalar biasa (bukan @)
    if n <= _BASE_SIZE:
        return _base_multiply(A, B)

    # Divide
    A11, A12, A21, A22 = _split(A)
    B11, B12, B21, B22 = _split(B)

    # Conquer — 7 perkalian rekursif Strassen
    M1 = _strassen_rec(_add(A11, A22), _add(B11, B22))
    M2 = _strassen_rec(_add(A21, A22), B11)
    M3 = _strassen_rec(A11,            _sub(B12, B22))
    M4 = _strassen_rec(A22,            _sub(B21, B11))
    M5 = _strassen_rec(_add(A11, A12), B22)
    M6 = _strassen_rec(_sub(A21, A11), _add(B11, B12))
    M7 = _strassen_rec(_sub(A12, A22), _add(B21, B22))

    # Combine
    C11 = _sub(_add(_add(M1, M4), M7), M5)
    C12 = _add(M3, M5)
    C21 = _add(M2, M4)
    C22 = _add(_sub(_add(M1, M3), M2), M6)

    return _join(C11, C12, C21, C22)

# ── entry point ───────────────────────────────────────────────────────────────
def multiply(A, B):
    # Cek dimensi
    n_A, p_A = len(A), len(A[0])
    p_B, m_B = len(B), len(B[0])
    
    if p_A != p_B:
        raise ValueError(f"Dimensi tidak cocok: A({n_A}x{p_A}) tidak bisa dikalikan B({p_B}x{m_B})")
    
    # Tentukan ukuran persegi power-of-2 terkecil yang menampung semua dimensi
    max_dim = max(n_A, p_A, m_B)
    size = 1
    while size < max_dim:
        size *= 2
    
    # Jika ukuran terlalu besar untuk Strassen, lemparkan error atau fallback ke NumPy
    if size > 8192:
        raise ValueError(f"Ukuran padding terlalu besar untuk Strassen: {size}×{size}")
    
    # Buat matriks A_pad dan B_pad (diisi 0 di area padding)
    A_pad = [[0.0] * size for _ in range(size)]
    B_pad = [[0.0] * size for _ in range(size)]
    
    # Salin A ke A_pad
    for i in range(n_A):
        for j in range(p_A):
            A_pad[i][j] = float(A[i][j])
    
    # Salin B ke B_pad
    for i in range(p_B):
        for j in range(m_B):
            B_pad[i][j] = float(B[i][j])
    
    # Panggil _strassen_rec dengan matriks persegi
    C_pad = _strassen_rec(A_pad, B_pad)
    
    # Crop hasil ke ukuran asli (n_A × m_B)
    return [C_pad[i][:m_B] for i in range(n_A)]