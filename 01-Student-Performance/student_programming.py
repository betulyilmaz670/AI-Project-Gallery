
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


# 2. adım
import matplotlib.pyplot as plt
import seaborn as sns



# ADIM 2 -- GÖRSELLEŞTİRME
# Amacımız:Veriyi gözle görmek
# Hangi faktörler notu etkiliyor


# 2.1 Matematik notlarının dağılımını gösteren histogram
# Notlar nasıl değişmiş bakalım
plt.figure(figsize=(10, 4))
sns.histplot(df['math score'], bins=20,color='blue',kde=True)
plt.title('Matematik Notlarının Dağılımı')
plt.xlabel('Matematik Notu')
plt.ylabel('Öprenci Sayısı')
plt.show()

# 2.2 Cinsiyete göre notlar
# kız ve erkek öğrencilerin notları arasında fark var mı?
plt.figure(figsize=(10, 4))
sns.boxplot(x='gender', y='math score', data=df)
plt.title('Cinsiyete Göre Matematik Notları')
plt.show()


# 2.3 Test hazırlık kursunun etkisi
# Kursa katılanlar başarılı mı
plt.figure(figsize=(10, 4))
sns.boxplot(x='test preparation course', y='math score', data=df)
# sns.histplot(df['math score'], bins=20,color='blue',kde=True) 
plt.title('Test Hazırlık Kursunun Notlara Etkisi')
plt.show()

# 2.4 Kolerasyon Haritası
# Hangi faktörler birbirine bağlı?
plt.figure(figsize=(8,6))
sns.heatmap(df[['math score', 'reading score', 'writing score']].corr(), annot=True, cmap='coolwarm',fmt='.2f')
plt.title('Notlar Arası Korelasyon')
plt.show()



