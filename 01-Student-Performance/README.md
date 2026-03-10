#  Student Performance Prediction

##  Proje Hakkında
Bu proje, öğrencilerin matematik, okuma ve yazma notlarını
etkileyen faktörleri analiz ederek makine öğrenimi ile
not tahmini yapmaktadır.

##  Veri Seti
- Kaynak: Kaggle - Students Performance in Exams
- 1000 öğrenci, 8 özellik
- Özellikler: cinsiyet, etnik köken, ebeveyn eğitim durumu,
  öğle yemeği, test hazırlık kursu, matematik/okuma/yazma notu

##  Proje Adımları
- [x] Veri yükleme ve EDA
- [x] Görselleştirme
- [x] Veri hazırlama (kategorik → sayısal)
- [x] Model kurma ve eğitme
- [x] Sonuçları değerlendirme

##  Kullanılan Modeller
| Model | R² Skoru | RMSE |
|-------|----------|------|
| Linear Regression | 0.87 | 5.20 |
| Decision Tree | 0.79 | 6.80 |
| Random Forest | 0.91 | 4.40 |

##  En İyi Model
- **Random Forest**
- R² : 0.91 → notların %91'ini doğru açıklıyor
- RMSE: 4.40 → tahminler ortalama 4.4 puan yanılıyor

##  Kullanılan Teknolojiler
- Python 3.x
- Pandas
- Scikit-learn
- Matplotlib & Seaborn
- NumPy

##  Nasıl Çalıştırılır?
pip install pandas scikit-learn matplotlib seaborn numpy
python student_programming.py