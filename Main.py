import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image

# 1. IMPLEMENTASI BRUTE FORCE (O(n^3))
def brute_force(A, B):
    n = len(A)
    C = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

# 2. IMPLEMENTASI STRASSEN (O(n^2.81))
def strassen(A, B):
    n = len(A)
    
    # Base case: jika ukuran matriks sudah kecil, gunakan perkalian biasa
    # Ini juga bisa menjadi 'Threshold' untuk optimasi hybrid
    if n <= 2: 
        return brute_force(A, B)
    
    # Membagi matriks menjadi 4 sub-matriks
    mid = n // 2
    a11 = A[:mid, :mid]; a12 = A[:mid, mid:]
    a21 = A[mid:, :mid]; a22 = A[mid:, mid:]
    
    b11 = B[:mid, :mid]; b12 = B[:mid, mid:]
    b21 = B[mid:, :mid]; b22 = B[mid:, mid:]
    
    # 7 Rumus P Strassen
    p1 = strassen(a11 + a22, b11 + b22)
    p2 = strassen(a21 + a22, b11)
    p3 = strassen(a11, b12 - b22)
    p4 = strassen(a22, b21 - b11)
    p5 = strassen(a11 + a12, b22)
    p6 = strassen(a21 - a11, b11 + b12)
    p7 = strassen(a12 - a22, b21 + b22)
    
    # Menggabungkan hasil menjadi matriks C
    c11 = p1 + p4 - p5 + p7
    c12 = p3 + p5
    c21 = p2 + p4
    c22 = p1 - p2 + p3 + p6
    
    # Menyusun kembali sub-matriks menjadi satu matriks utuh
    C = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))
    return C

# 3. FUNGSI UTAMA UNTUK ANALISA
def analisa_algoritma(image_path):
    # Ukuran n yang akan diuji (harus pangkat 2)
    sizes = [16, 32, 64, 128] # Tambah ke 256 jika laptop kuat
    
    time_brute = []
    time_strassen = []
    
    img_asli = Image.open(image_path).convert('L') # Load ke 8-bit (Grayscale)

    print(f"{'n':<10} | {'Brute Force (s)':<18} | {'Strassen (s)':<15}")
    print("-" * 50)

    for n in sizes:
        # Resize citra ke n x n
        img_resized = img_asli.resize((n, n))
        A = np.array(img_resized, dtype=int)
        B = np.eye(n, dtype=int) # Matriks identitas sebagai pengali
        
        # Hitung Waktu Brute Force
        start = time.time()
        brute_force(A, B)
        end = time.time()
        t_brute = end - start
        time_brute.append(t_brute)
        
        # Hitung Waktu Strassen
        start = time.time()
        strassen(A, B)
        end = time.time()
        t_strassen = end - start
        time_strassen.append(t_strassen)
        
        print(f"{n:<10} | {t_brute:<18.5f} | {t_strassen:<15.5f}")

    # 4. MEMBUAT GRAFIK ANALISA
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, time_brute, label='Brute Force O(n^3)', marker='o', color='red')
    plt.plot(sizes, time_strassen, label='Strassen O(n^2.81)', marker='s', color='blue')
    
    plt.title('Perbandingan Waktu Eksekusi: Brute Force vs Strassen')
    plt.xlabel('Ukuran Matriks (n x n)')
    plt.ylabel('Waktu (detik)')
    plt.legend()
    plt.grid(True)
    plt.show()

# JALANKAN PROGRAM HALO
# Ganti 'foto_anda.jpg' dengan nama file gambar yang ada di folder yang sama
try:
    analisa_algoritma('foto_anda.jpg') 
except FileNotFoundError:
    print("Error: File gambar tidak ditemukan. Pastikan nama file benar.")
