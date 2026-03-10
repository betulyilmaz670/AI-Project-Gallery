
import pandas as pd
df=pd.read_csv('StudentsPerformance.csv')
print("=== VERİ SETİ ===")
print(df.head())  # İlk 5 satırı gösterir





# ---- 2.2 Kaç satır, kaç sütun var? ----
# (satır, sütun) şeklinde gösterir
print("\n=== BOYUT ===")
print(f"Satır sayısı  : {df.shape[0]}")
print(f"Sütun sayısı  : {df.shape[1]}")

# ---- 2.3 Sütun isimleri ve veri tipleri ----
# object = metin, int64 = tam sayı, float64 = ondalıklı
print("\n=== SÜTUNLAR VE TİPLER ===")
print(df.dtypes)

# ---- 2.4 Eksik veri var mı? ----
# 0 çıkarsa eksik veri yok demek — harika!
print("\n=== EKSİK VERİ KONTROLÜ ===")
print(df.isnull().sum())

# ---- 2.5 Sayısal sütunların istatistikleri ----
# count, mean, std, min, max gibi değerleri gösterir
print("\n=== İSTATİSTİKLER ===")
print(df.describe())
