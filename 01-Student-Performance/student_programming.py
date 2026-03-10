
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




# 3. adım
# ADIM 3 -- Veriyi Modelleme İçin Hazırlama

# 3.1 Sütun isimlerini görelim
print("Sütunlar:", df.columns.tolist())

# 3.2 Kategorik değişkenleri sayısal hale getirelim
df_encoded = pd.get_dummies(df, drop_first=True)

print("\n=== ENCODE EDİLMİŞ VERİ ===")
print(df_encoded.columns.tolist())

print("\n=== YENİ BOYUT ===")
print(f"Satır sayısı  : {df_encoded.shape[0]}")
print(f"Sütun sayısı  : {df_encoded.shape[1]}")

# 3.3 Hedef değişkeni tanımlayalım
# veriyi eğiteceğiz
X=df_encoded.drop(['math score'], axis=1)  # Bağımsız değişkenler
y=df_encoded['math score']  # Hedef değişken

print("\n=== BAĞIMSIZ DEĞİŞKENLER ===")
print(X.columns.tolist()),print(X.shape)
print("\n=== HEDEF DEĞİŞKEN ===")
print(y.name),print(y.shape)

# 3.4 Train/Test olarak Böl
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   
print("\n=== EĞİTİM VE TEST SETLERİ ===")
print(f"X_train boyutu: {X_train.shape[0]} öğrenci")
print(f"Test seti : {X_test.shape[0]} öğrenci")



# ADIM 4 -- MODEL KUR VE EĞİT
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# 4.1 Modeli tanımlayalım
# Her modeli bir sözcükte tutacağım
# Böylece tek bir döngüde test edebileceğiz
modeller = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

# 4.2 Her modeli eğitelim ve tahmin yapalım
sonuclar = {}
for isim, model in modeller.items():
    model.fit(X_train, y_train)  # Modeli eğit
    y_pred = model.predict(X_test)  # Test seti üzerinde tahmin yap
    r2 = r2_score(y_test, y_pred)  # R² skorunu hesapla
    rsme= np.sqrt(mean_squared_error(y_test, y_pred))  # RSME hesapla
    sonuclar[isim] = {'R²': round(r2, 4), 'RSME': round(rsme, 4)}  # Sonuçları kaydet
   
   
    print(f"{'='*40}")
    print(f"Model: {isim}")
    print(f"R² Score: {r2:.4f} ne kadar yakınsa o kadar iyi")
    print(f"RSME: {rsme:.4f} ne kadar küçükse o kadar iyi")


    # 4.3 En iyi modeli bul 
en_iyi=max(sonuclar, key=lambda x: sonuclar[x]['R²'])
print(f"n{'='*40}")
print(f"En İyi Model: {en_iyi}")
print(f"R²: {sonuclar[en_iyi]['R²']}")
print(f"RSME: {sonuclar[en_iyi]['RSME']}")


# ADIM 5 -- SONUÇLARI DEĞERLENDİRELİM
# Amacımız:en iyi modeli seçip tahminleri görsel olarak karşılaştırmak

# 5.1 Model karşılaştırma tablosu
print("\n=== MODEL KARŞILAŞTIRMASI ===")
print(f"{'Model':<25} {'R²':>8} {'RSME':>8}")
print("-"*45)
for isim, skorlar in sonuclar.items():
    print(f"{isim:<25} {skorlar['R²']:>8.4f} {skorlar['RSME']:>8.4f}")

# 5.2 en iyi modelleme tahmin grafiği
# Gerçek değerler vs Tahmin edilen değerler
en_iyi_model = modeller[en_iyi]
y_pred_en_iyi = en_iyi_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_en_iyi, color='blue', alpha=0.5)

# Mükemmel tahmin çizgisi-bütüj noktalar burada olsaydı model mükemmeldi
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2, label='Mükemmel Tahmin')

plt.xlabel('Gerçek Notlar')
plt.ylabel('Tahmin Edilen Notlar')
plt.title(f'{en_iyi} Tahminleri vs Gerçek Notlar')
plt.legend()
plt.show()

# 5.3 Random Forest -- önemli ozellikler
# Hangi özellik notu en çok etkiliyor
rf_model = modeller['Random Forest']

plt.figure(figsize=(10, 6))
onem=pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)

onem.plot(kind='barh', color='green')
plt.title('Notu en çok etkileyen özellikler')
plt.xlabel('Önem skoru')
plt.tight_layout()
plt.show()


# 5.4 Örnek tahmin yap
# Yeni öğrenci için not tahmini
print("\n=== ÖRNEK TAHMİN ===")
yeni_ogrenci = X_test.iloc[0:1]  # Test setinden bir öğrenci alalım
tahmin = en_iyi_model.predict(yeni_ogrenci)
gercek = y_test.iloc[0]

print(f"Gerçek Not: {gercek:.1f}")
print(f"Tahmin Edilen Not: {tahmin[0]:.1f}")
print(f"Fark: {abs(gercek - tahmin[0]):.1f} puan")
