# UÇUŞ ÜCRETİ TAHMİNİ
# Amacımız:Uçuş öz. göre bilet fiyatını tahmin etmek

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1.1Veriyi Yükleme
df=pd.read_excel("Data_Train.xlsx")

# 1.2 İlk Bakış
print("Veri Setinin İlk 5 Satırı:")
print(df.head())
print("\nVeri Setinin Boyutu:")
print(f"Satır Sayısı: {df.shape[0]}, Sütun Sayısı: {df.shape[1]}")

print("Sütunlar ve Tipler")
print(df.dtypes)

print("Eksik Değerler:")
print(df.isnull().sum())

print("Temel İstatistikler:")
print(df.describe())

# ADIM 2-VERİ TEMİZLEME
# Bu projede tarih ve saat sütunlatı var bunları çevireceğiz
# Hepsini sayı yapacağım çünkü modelimiz 14:10 gibi şeyleri anlayamaz

# 2.1 Eksik Verileri Sil
df.dropna(inplace=True)
print(f"Temizleme sonrası satır sayısı: {df.shape[0]}")

# 2.2 Tarihten gün ve ay çıkarma
df['Journey_day'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y').dt.day
df['Journey_month'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y').dt.month

# Orjinal tarih sütuununu silelim
df.drop('Date_of_Journey', axis=1, inplace=True)

print("\n TARİH SÜTUNLARI")
print(df[['Journey_day', 'Journey_month']].head())

# 2.3 Kalkıi saatinden saat ve dk yı çıkaralım
df['Dep_hour'] = pd.to_datetime(df['Dep_Time']).dt.hour
df['Dep_min'] = pd.to_datetime(df['Dep_Time']).dt.minute
df.drop('Dep_Time', axis=1, inplace=True)

# 2.4 Varış saatinden saat ve dk yı çıkaralım
df['Arrival_hour'] = pd.to_datetime(df['Arrival_Time']).dt.hour
df['Arrival_min'] = pd.to_datetime(df['Arrival_Time']).dt.minute
df.drop('Arrival_Time', axis=1, inplace=True)

print("\n SAAT SÜTUNLARI")
print(df[['Dep_hour', 'Dep_min', 'Arrival_hour', 'Arrival_min']].head())

# 2.5 Uçuş süresini dakika cinsinden hesaplayalım
def sure_dakika(sure):
    sure=str(sure)
    saat=0
    dakika=0
    if 'h' in sure:
        saat=int(sure.split('h')[0])
    if 'm' in sure:
        dakika=int(sure.split('m')[0].split()[-1])
    return saat*60+dakika

df['Duration_mins'] = df['Duration'].apply(sure_dakika)
df.drop('Duration', axis=1, inplace=True)

print("\n UÇUİŞ SÜRESİ SÜTUNU")
df['Total_Stops'] = df['Total_Stops'].map({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})

print("\n AKTARMA SAYISI")
print(df['Total_Stops'].value_counts())


# ADIM 3 -KATEGORİK VERİLERİ SAYISALA ÇEVİRME
# metin içeren sütunları sayısala çevirelim

# 3.1 Gereksiz sütunları silelim
df.drop(['Route', 'Additional_Info'], axis=1, inplace=True)

print("\n KALAN SÜTUNLAR")
print(df.columns.tolist())

# 3.2 Kategorik sütunları görelim
katekorik=['Airline', 'Source', 'Destination']
for sutun in katekorik:
    print(f"\n {sutun} Sütunu Değerleri:")
    print(df[sutun].value_counts())

# 3.3 One-Hot Encoding uygulayalım
# her katekori için yeni sütunlar oluşturalım
df_encoded = pd.get_dummies(df, columns=katekorik, drop_first=True)

print("\n One-Hot Encoding Sonrası Sütunlar")
print(f"\n Toplam Sütun Sayısı: {df_encoded.shape[1]}")
print(df_encoded.columns.tolist())

# 3.4 Hedef değişkeni ve özellikleri ayıralım
# X=modelin bakacağı özellikler
# y=tahmin edeceğimiz fiyat
X = df_encoded.drop('Price', axis=1)
y = df_encoded['Price']

print(f"\nX boyutu: {X.shape}")
print(f"\nY boyutu: {y.shape}")

# 3.5 Train/Test böl
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   
print(f"\nEğitim seti boyutu: {X_train.shape[0]} uçuş")
print(f"\nTest seti boyutu: {X_test.shape[0]} uçuş")

# ADIM 4-MODEL KURULUMU  VE EĞİTME
# aynı formülü kullanacağız ama rakamlar daha büyük olacak
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np

# 4.1 modeli tanımlayalım
modeller = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)

}

# 4.2 modelleri eğitelim ve değerlendirelim
sonuclar = {}
for isim, model in modeller.items():
#    Eğitelim
    model.fit(X_train, y_train)
#    Tahmin yapalım
    y_pred = model.predict(X_test)
#   metrikleri hesaplayalım
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    sonuclar[isim] = {'R2': round(r2, 4), 'RMSE': round(rmse, 4), 'RMSE': round(rmse, 4)}

    print(f"\n{'='*45}")
    print(f"Model : {isim}")
    print(f"R²    : {r2:.4f}  → 1e yakın olması iyi")
    print(f"RMSE  : {rmse:.2f} → fiyat tahmini ortalama bu kadar yanılıyor")

# 4.3 en iyi modeli seçelim
en_iyi=max(sonuclar, key=lambda x: sonuclar[x]['R2'])
print(f"\n{'='*45}")
print(f"🏆 En iyi model : {en_iyi}")
print(f"   R²           : {sonuclar[en_iyi]['R2']}")
print(f"   RMSE         : {sonuclar[en_iyi]['RMSE']} ₹")

   
# ADIM 5-SONUÇLARI GÖRSELLEŞTİRME

# 5.1 Fiyat dağılımı
# Fiyatlar nasıl dağılmş görelim
plt.figure(figsize=(10, 4))
sns.histplot(df_encoded['Price'], bins=50,color='steelblue', kde=True)
plt.title('Bilet Fiyatlarının Dağılımı')
plt.xlabel('Fiyat (₹)')
plt.ylabel('Uçuş Sayısı')
plt.show()

# 5.2 Havayoluna göre fiyat
plt.figure(figsize=(12, 6))
sns.boxplot(x='Airline', y='Price', data=df)
plt.xticks(rotation=45)
plt.title('Havayoluna Göre Bilet Fiyatları')
plt.tight_layout()
plt.show()

# 5.3 Aktatma sayısına göre fiyat
# Daha fazla aktarma ucuz mu?
plt.figure(figsize=(8, 5))
sns.boxplot(x='Total_Stops', y='Price', data=df)
plt.title('Aktarma Sayısına Göre Bilet Fiyatları')
plt.xlabel('Aktarma Sayısı')
plt.ylabel('Fiyat (₹)')
plt.show()

# 5.4 Gerçek vs Tahmin
en_iyi_model = modeller[en_iyi]
y_pred_en_iyi = en_iyi_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_en_iyi, alpha=0.3, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red',linewidth=2,label='mükemmel tahmin')  # 45 derece çizgi
plt.xlabel('Gerçek Fiyat (₹)')
plt.ylabel('Tahmin Edilen Fiyat (₹)')
plt.title(f'Gerçek vs Tahmin - {en_iyi}')
plt.legend()
plt.show()

# 5.5 En Önemli Özellikler
# Fiyatı en çok etkileyen faktörler
rf_model = modeller['Random Forest']

plt.figure(figsize=(12, 8))

onem=pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True).tail(15)
onem.plot(kind='barh', color='steelblue')
plt.title('Fiyatı En Çok Etkileyen 15 Özellik')
plt.xlabel('Özellik Önemi')
plt.tight_layout()
plt.show()


# 5.6 Örnek Tahmin
print("\nÖrnek Tahmin:")
yeni_ucus=X_test.iloc[0:1]  # Test setinden bir uçuş alalım
tahmin=en_iyi_model.predict(yeni_ucus)
gercek=y_test.iloc[0]

print(f"Gerçek Fiyat: {gercek} ₹")
print(f"Tahmin Edilen Fiyat: {tahmin[0]:.2f} ₹")
print(f"Fark: {abs(gercek - tahmin[0]):.0f} ₹")

