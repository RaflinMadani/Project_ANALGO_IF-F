import cv2
import os

jalur_asli = "image/Lena_Warna.jpg"
img_asli = cv2.imread(jalur_asli)

if img_asli is None:
    print(f"Error: File '{jalur_asli}' tidak ditemukan!")
else:
    lebar, tinggi = 1024, 1024
    ukuran_baru = (lebar, tinggi)

    img_resolution = cv2.resize(img_asli, ukuran_baru, interpolation=cv2.INTER_CUBIC)
    nama_file_saja = os.path.splitext(os.path.basename(jalur_asli))[0]
    nama_file_baru = f"{nama_file_saja}_{lebar}x{tinggi}.jpg"
    
    cv2.imwrite(nama_file_baru, img_resolution)
    print("Mantap! Gambar berhasil di-resize.")
    print(f"File disimpan di folder utama dengan nama: {nama_file_baru}")