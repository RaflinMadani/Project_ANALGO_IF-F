def multiply(A: list, B: list) -> list:
    n = len(A)
    p = len(A[0])
    m = len(B[0])

    if len(B) != p:
        raise ValueError(
            f"Dimensi tidak cocok: A({n}x{p}) tidak bisa dikalikan B({len(B)}x{m})"
        )

    C = [[0.0] * m for _ in range(n)]

    # Loop reorder: i → k → j  (cache-friendly)
    for i in range(n):
        for k in range(p):
            a_ik = A[i][k]          # prefetch baris A — baca sekali untuk seluruh loop j
            for j in range(m):
                C[i][j] += a_ik * B[k][j]   # B[k][j] row-major, sequential

    return C