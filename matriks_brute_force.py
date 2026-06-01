def multiply(A: list, B: list) -> list:
    n = len(A)
    p = len(A[0])
    m = len(B[0])

    if len(B) != p:
        raise ValueError(
            f"Dimensi tidak cocok: A({n}x{p}) tidak bisa dikalikan B({len(B)}x{m})"
        )

    # Inisialisasi matriks hasil dengan nol
    C = [[0.0] * m for _ in range(n)]

    # Triple nested loop — i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                C[i][j] += A[i][k] * B[k][j]

    return C