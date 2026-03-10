# AIRBNB FİYAT TAHMİNİ
# Amacımız:Airbnb ilanlarının özelliklerine göre fiyat tahmini yapmak.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1.1 veriyi yükleyelim
df=pd.read_csv('AB_NYC_2019.csv')

# 1.2 ilk bakış
print("ilk 5 satır:")
print(df.head())

print("Boyut:")
print(f"Satır sayısı: {df.shape[0]}, Sütun sayısı: {df.shape[1]}")

print(f"\Sütunlar ve veri tipleri:")
print(df.dtypes)

print("\n Eksik değerler:")
print(df.isnull().sum())

print("\n Temel istatistikler:")
print(df.describe())

# Adım 2 -Veri Temizleme

# 2.1 Gereksiz Sütunları Kaldırma
df.drop(['id', 'name', 'host_id', 'host_name', 'last_review'], axis=1, inplace=True)
print("kalan sütunlar:")
print(df.columns.tolist())

# 2.2 Eksik verileri doldurma
df['reviews_per_month'].fillna(0, inplace=True)

print("\n Eksik veri kontrolü:")
print(df.isnull().sum())

# 2.3 Fiyat dağılımını inceleyelim
print("\n Fiyat İstatikleri:")
print(f"Minimum Fiyat: {df['price'].min()}")
print(f"Maksimum Fiyat: {df['price'].max()}")
print(f"Ortalama Fiyat: {df['price'].mean()}")
print(f"Medyan Fiyat: {df['price'].median()}")

# Görselleştirme
plt.figure(figsize=(10, 4))
sns.histplot(df['price'], bins=100, color='steelblue')
plt.title('Fiyat Dağılımı')
plt.xlabel('Fiyat')
plt.show()

# 2.4 Aykırı Değerleri temizleyelim
print(f"\n Temizleme öncesi: {df.shape[0]} ilan")

df=df[df['price'] > 0]
df=df[df['price'] < 1000]

# Temizlenmiş fiyat dağılımı
plt.figure(figsize=(10, 4))
sns.histplot(df['price'], bins=100, color='green')
plt.title('Temizlenmiş Fiyat Dağılımı')
plt.xlabel('Fiyat')
plt.show()

# Adım 3-Veriyi Hazılayalım
print("\n Kategorik ve sayısal sütunlar:")
print(f"nneighborhood_group değerleri:")
print(df['neighbourhood_group'].value_counts())

print(f"\nroom_type değerleri:")
print(df['room_type'].value_counts())

# 3.2 Neighbourhood sütununu silelim
# çok fazla değer var ve bu grup orayı temsil ediyor
df.drop('neighbourhood', axis=1, inplace=True)

df.encoded = pd.get_dummies(df,columns=['neighbourhood_group','room_type'], drop_first=True)

print("\n Dönüşüm sonrası sütunlar:")
print(df.encoded.columns.tolist())

# 3.4 Hedef ve özellikleri ayıralım
X=df.encoded.drop('price', axis=1)
y=df.encoded['price']

print(f"nX boyutu: {X.shape}")
print(f"y boyutu: {y.shape}")

# 3.5 Train/Test Bölümü
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"nEğitim seti boyutu: {X_train.shape[0]} ilan")
print(f"Test seti boyutu: {X_test.shape[0]} ilan")



# ADIM 4 — MODELİ KUR VE EĞİT



from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# 4.1 Modelleri tanımla 
modeller = {
    'Linear Regression' : LinearRegression(),
    'Decision Tree'     : DecisionTreeRegressor(random_state=42),
    'Random Forest'     : RandomForestRegressor(n_estimators=100,
                                                random_state=42)
}

#  4.2 Her modeli eğit ve değerlendirelim 
sonuclar = {}

for isim, model in modeller.items():
    print(f"\n{isim} eğitiliyor...")

    # Eğitelim
    model.fit(X_train, y_train)

    # Tahmin etme
    y_pred = model.predict(X_test)

    # Metrikleri hesaplayalım
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    sonuclar[isim] = {'R2': round(r2, 4), 'RMSE': round(rmse, 2)}

    print(f"R²   : {r2:.4f}")
    print(f"RMSE : {rmse:.2f} $")

#  4.3 En iyi modeli bulalım
en_iyi = max(sonuclar, key=lambda x: sonuclar[x]['R2'])

print(f"\n{'='*45}")
print(f" En iyi model : {en_iyi}")
print(f"   R²           : {sonuclar[en_iyi]['R2']}")
print(f"   RMSE         : {sonuclar[en_iyi]['RMSE']} $")


# ADIM 5 — SONUÇLARI GÖRSELLEŞTİR


# 5.1 Bölgeye göre fiyat
# Hangi bölge daha pahalı?
plt.figure(figsize=(10, 5))
sns.boxplot(x='neighbourhood_group', y='price', data=df)
plt.title('Bölgeye Göre Fiyat')
plt.xlabel('Bölge')
plt.ylabel('Fiyat ($)')
plt.show()

#  5.2 Oda tipine göre fiyat 
# Tüm ev mi, özel oda mı daha pahalı?
plt.figure(figsize=(8, 5))
sns.boxplot(x='room_type', y='price', data=df)
plt.title('Oda Tipine Göre Fiyat')
plt.xlabel('Oda Tipi')
plt.ylabel('Fiyat ($)')
plt.show()

# 5.3 Korelasyon haritası
# Sayısal sütunlar arasındaki ilişki
plt.figure(figsize=(10, 8))
sayisal = df[['price', 'minimum_nights', 'number_of_reviews',
              'reviews_per_month', 'calculated_host_listings_count',
              'availability_365']]
sns.heatmap(sayisal.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korelasyon Haritası')
plt.show()

#  5.4 Gerçek vs Tahmin
en_iyi_model = modeller[en_iyi]
y_pred_en_iyi = en_iyi_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_en_iyi, color='blue', alpha=0.3)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linewidth=2, label='Mükemmel Tahmin')
plt.xlabel('Gerçek Fiyat ($)')
plt.ylabel('Tahmin Edilen Fiyat ($)')
plt.title(f'Gerçek vs Tahmin — {en_iyi}')
plt.legend()
plt.show()

# 5.5 En önemli özellikler
rf_model = modeller['Random Forest']

plt.figure(figsize=(10, 6))
onem = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=True).tail(10)

onem.plot(kind='barh', color='steelblue')
plt.title('Fiyatı En Çok Etkileyen 10 Özellik')
plt.xlabel('Önem Skoru')
plt.tight_layout()
plt.show()

# 5.6 Örnek tahmin edelim 
print("\n=== ÖRNEK TAHMİN ===")
yeni_ilan = X_test.iloc[0:1]
tahmin = en_iyi_model.predict(yeni_ilan)
gercek = y_test.iloc[0]

print(f"Gerçek fiyat : {gercek:.0f} $")
print(f"Tahmin       : {tahmin[0]:.0f} $")
print(f"Fark         : {abs(gercek - tahmin[0]):.0f} $")

#  5.7 Model karşılaştırma tablosu 
print(f"\n{'='*45}")
print(f"{'Model':<25} {'R²':>8} {'RMSE':>8}")
print("-" * 45)
for isim, metrik in sonuclar.items():
    print(f"{isim:<25} {metrik['R2']:>8} {metrik['RMSE']:>8}")
