import numpy as np

def multiply(A: list, B: list) -> list:
    A_np = np.array(A, dtype=np.float64)
    B_np = np.array(B, dtype=np.float64)

    if A_np.shape[1] != B_np.shape[0]:
        raise ValueError(
            f"Dimensi tidak cocok: A({A_np.shape[0]}x{A_np.shape[1]}) "
            f"tidak bisa dikalikan B({B_np.shape[0]}x{B_np.shape[1]})"
        )

    C_np = A_np @ B_np

    return C_np.tolist()