"""
Analisis Regresi Linear Berganda untuk Memprediksi Jumlah Siswa Sekolah Menengah Atas 
di Kota Palu Berdasarkan Ketersediaan Sekolah dan Jumlah Guru Tingkat Kecamatan

Metode: Multiple Linear Regression (Machine Learning)
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    print("="*70)
    print(" MEMULAI PROGRAM MACHINE LEARNING: REGRESI LINEAR BERGANDA")
    print("="*70)

    # 1. Memuat Dataset Kecil (Nama file sudah disesuaikan dengan file barumu)
    file_name = 'DATASET_SMA PALU.xlsx - Dataset Kecil.csv'
    try:
        dataset = pd.read_csv(file_name)
        print(f"[INFO] Dataset '{file_name}' berhasil dimuat.")
        print(f"[INFO] Total observasi data: {len(dataset)} baris.\n")
    except FileNotFoundError:
        print(f"[ERROR] File '{file_name}' tidak ditemukan. Pastikan file ada di folder yang sama.")
        return

    # 2. Mendefinisikan Variabel Independen (X) dan Dependen (Y)
    # Deteksi otomatis nama kolom Y (berjaga-jaga jika menggunakan 'Siswa' atau 'Murid')
    kolom_y = 'Jumlah Siswa (Y)' if 'Jumlah Siswa (Y)' in dataset.columns else 'Jumlah Murid (Y)'
    
    X = dataset[['Jumlah Sekolah (X1)', 'Jumlah Guru (X2)']]
    Y = dataset[kolom_y]

    # 3. Inisialisasi dan Pelatihan Model Machine Learning
    print("[INFO] Melatih model Regresi Linear Berganda...")
    model_regresi = LinearRegression()
    model_regresi.fit(X, Y)

    # 4. Mendapatkan Parameter Matematika
    konstanta_b0 = model_regresi.intercept_
    koefisien_b1 = model_regresi.coef_[0]
    koefisien_b2 = model_regresi.coef_[1]

    # 5. Prediksi dan Akurasi
    prediksi_y = model_regresi.predict(X)
    nilai_r2 = r2_score(Y, prediksi_y)
    nilai_mse = mean_squared_error(Y, prediksi_y)

    # 6. Menampilkan Hasil untuk Laporan Dosen
    print("\n" + "="*70)
    print(" HASIL ANALISIS STATISTIK DAN EVALUASI MODEL")
    print("="*70)
    print(f"Persamaan Regresi : Y = {konstanta_b0:.2f} + ({koefisien_b1:.2f} * X1) + ({koefisien_b2:.2f} * X2)")
    print(f"Konstanta (B0)    : {konstanta_b0:.4f}")
    print(f"Koefisien X1 (B1) : {koefisien_b1:.4f} (Pengaruh Jumlah Sekolah)")
    print(f"Koefisien X2 (B2) : {koefisien_b2:.4f} (Pengaruh Jumlah Guru)")
    print("-" * 70)
    print(f"Akurasi Model (R-Squared) : {nilai_r2:.4f} ({(nilai_r2*100):.2f}%)")
    print(f"Mean Squared Error (MSE)  : {nilai_mse:.2f}")
    print("="*70)

    # 7. Menampilkan Grafik
    plt.figure(figsize=(10, 6))
    plt.scatter(Y, prediksi_y, color='blue', alpha=0.6, label='Titik Data Prediksi')
    plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='red', linewidth=2, label='Garis Linear Ideal')
    plt.title('Grafik Aktual vs Prediksi Jumlah Siswa SMA di Kota Palu', fontsize=14)
    plt.xlabel('Jumlah Siswa Aktual (Data BPS)', fontsize=12)
    plt.ylabel('Jumlah Siswa Hasil Prediksi Model', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig('grafik_regresi_palu.png')
    print("\n[INFO] Grafik visualisasi berhasil disimpan sebagai 'grafik_regresi_palu.png'.")
    plt.show()

if __name__ == '__main__':
    main()