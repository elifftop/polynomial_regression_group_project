# polynomial_regression_group_project
Proje Başlığı
Polynomial Regression

Açıklama
Polinomsal Regresyon ile Python Uygulaması projesinde; hava durumu veri setindeki
değişkenlerden yararlanarak Nem oranı değişkeni tahmin edilmiştir. Polinom derecesi ve
polinom özellikleri belirlendikten sonra regresyon modeli eğitilmiş ve tahminler yapılmıştır.
Elde edilen sonuçlar, tahminlerin doğruluğunu ve modelin performansını değerlendirmek için
istatistiksel metriklerle (MSE,RMSE, R2) birlikte sunulmuştur. Polinomsal Regresyon ile
eğitilen model Lineer Regresyon, Catboost Regressor ve LightGBM Regressor ile
kıyaslanmıştır. Sonuçta en iyi tahmini karar ağaçları tabanlı Catboost Regressor ve
LightGBM Regressor modelleri ardından Polinom Regresyon modeli gerçekleştirmiştir. Bu
proje Ecem ÇEŞMECİLER, Elif TOP, Erdem ARDUÇ, Hüseyin MURAT, Utku Sait ÇİÇEK,
Yunus YILDIRIM tarafından 15 HAZİRAN – 22 HAZİRAN arasında geliştirilmiştir.

Kullanılan Kütüphaneler
!pip install catboost
!pip install lightgbm
!pip install gdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
import glob
import os
import shutil

Veri Seti
Projede kullanılan veri setini Google Drive'dan indirme işlemi burada yapılıyor.
!gdown --id 1ZOvwyjxtQYa3VNjECSZCQyEYqoR2tDXb
!gdown --id 1wKejL9j0QPmV7oQLkGvMkJ4gJG3X_BOI
!gdown --id 1oSiWumalsW_KGsRcCAqhaBWMl8DNaE9B
!gdown --id 1CRCm1jn13JfdkpdFZ4sZjt1kqsxoko0x
!gdown --id 10jYtttcrrDcYKl-NJBFfJGcU0HDPQGEz
!gdown --id 1QEhm29Cmm4knAnBtPmaKJeb_sRsVuALj
!gdown --id 18bKqz7K-2JoO5bhPIypaNXMC_eZddlLa
!gdown --id 1PC71NMkNCRAUcIzt34Y6nlKj1q0WD3wg
!gdown --id 1zE0lU883p5o8SZzMA-3Dety8NWBkKqsB
!gdown --id 1n2_zocTJO5zJzjva4ENv7e_RzkUlqKJz
!gdown --id 1qRcliOb56t7ip5copHTPGousKGFnLx4n
!gdown --id 13TSa0wNJ8BiqBQ5fCBOdkyBtyExQ1Yuu

Klasör Oluşturma ve Dosya Taşıma
Bu bölümde kod, gorsel adında bir klasör oluşturur ve .png uzantılı tüm dosyaları bu klasöre taşır.
# 'gorsel' adında bir klasör oluştur
os.makedirs('gorsel', exist_ok=True)

# .png uzantılı tüm dosyaları bul
png_files = glob.glob('*.png')

# Her dosyayı yeni klasöre taşı
for file in png_files:
    shutil.move(file, 'gorsel/')

Veri Ön Hazırlık ve Temizleme
Bu bölümde, projenizde kullanılan hava durumu veri setini temizlemek ve eksik değerleri analiz etmek için kullanılan Python kodlarını bulacaksınız.

Veri Seti
Önceki kod, `testset.csv` adında bir veri setini okur ve bu veri setiyle çalışmayı sağlar.

df = pd.read_csv("testset.csv")
df.info()

Hava Durumu Veri Seti Kolon Kısaltmalarının Açıklamaları
Veri setinde kullanılan kısaltmalar ve bunların açıklamaları aşağıdaki gibidir:

_dewptm: Çiğ Noktası Sıcaklığı
_fog: Sis
_hail: Dolu
_hum: Nem
_pressurem: Basınç
_rain: Yağmur
_snow: Kar
_tempm: Sıcaklık
_thunder: Gökgürültüsü
_tornado: Tornado
_vism: Görüş Mesafesi
_wspdm: Rüzgar Hızı

Eksik Değer Analizi
Veri setindeki eksik değerleri analiz etmek için aşağıdaki Python kodları kullanılmıştır:
def eksik_deger_tablosu(df):
    eksik_deger = df.isnull().sum()
    eksik_deger_yuzde = 100 * df.isnull().sum() / len(df)
    eksik_deger_tablo = pd.concat([eksik_deger, eksik_deger_yuzde], axis=1)
    eksik_deger_tablo_son = eksik_deger_tablo.rename(
        columns={0: 'Eksik Değerler', 1: '% Değeri'})
    return eksik_deger_tablo_son

eksik_deger_tablosu(df)

Ayrıca eksik değerlerin ısı haritasını görselleştirmek için şu kodları kullanabilirsiniz:
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df.isnull().sum().to_frame(), cmap='BuPu')
plt.title("Eksik Değerler Isı Haritası")
plt.show()

sns.heatmap((100 * df.isnull().sum() / len(df)).to_frame(), cmap='BuPu')
plt.title("Eksik Değerler Yüzde Isı Haritası")
plt.show()

Korelasyon Analizi ve Görselleştirme
Bu bölümde, veri kaybı analizi ve korelasyon analizi için kullanılan Python kodlarına ve sonuçlarına yer verilmiştir.

Veri Kaybı Analizi
Veri setindeki eksik değerlerin sayısını ve görünüşünü renk skalası kullanarak görmek için aşağıdaki kod kullanılmıştır:
import seaborn as sns
import matplotlib.pyplot as plt

# Renk skalası ile eksik veri sayısını göster
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap='coolwarm', cbar=False)
plt.title("Eksik Veri Renk Skalası")
plt.show()
Açık mavi renkteki alanlar eksik veri olmadığını, mor renkteki alanlar ise eksik veri olduğunu göstermektedir.

Veri Seti Temizleme
Veri kaybının %14 ve üzerinde olduğu kolonlar aşağıdaki kod kullanılarak veri setinden çıkarılmıştır:

df.drop(["datetime_utc"," _windchillm"," _wgustm"," _wdire"," _wdird"," _precipm"," _heatindexm"], axis=1, inplace=True)

Hava Şartları Doldurma
Veri setindeki hava şartları kolonundaki eksik değerler en çok tekrar eden "Haze" değeri ile doldurulmuştur:
most_common_condition = df[" _conds"].value_counts().idxmax()
df[" _conds"] = df[" _conds"].fillna(most_common_condition)
Kategorik Verileri Dönüştürme
Son olarak, kategorik veriler "get_dummies" yapısı kullanılarak dönüştürülmüş ve orijinal kolon çıkarılmıştır:
df_encoded = pd.get_dummies(df[' _conds'], prefix='encoded')
df = pd.concat([df, df_encoded], axis=1).reindex(df.index)
df.drop(' _conds', axis=1, inplace=True)
Korelasyon Analizi ve Görselleştirme
Veri setinin korelasyonları aşağıdaki ısı haritası ile görselleştirilmiştir:
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='BuPu', center=0)
plt.title("Korelasyon Isı Haritası")
plt.show()
Bu ısı haritası, veri setinin sayısal değişkenleri arasındaki ilişkileri görsel olarak gösterir ve verileri daha iyi anlamanıza yardımcı olabilir.

Veri Temizleme Devamı
Bu bölümde, veri setinin daha fazla temizlenmesi ve aykırı değerlerin işlenmesi için kullanılan Python kodlarına ve sonuçlarına yer verilmiştir.

NaN Değerleri Kaldırma
`corr()` yöntemi ile elde edilen korelasyon matrisinde NaN değerleri içeren kolonlar aşağıdaki kod kullanılarak kaldırılmıştır:
df.drop("encoded_Light Freezing Rain", axis=1, inplace=True)

Korelasyon Bazlı Kolon Eleme
Kolonlar arasındaki düşük korelasyonlu olanlar, nem (_hum) ile arasındaki korelasyon değeri 0.05'ten düşük olanlar, aşağıdaki kod kullanılarak kaldırılmıştır:
column_list = df.columns.tolist()
for i in column_list:
    if (abs(df[" _hum"].corr(df[i])) < 0.05):
        df.drop(i, axis=1, inplace=True)

Korelasyon Görselleştirme
Kolonlar arasındaki korelasyon, aşağıdaki görselleştirme ile gösterilmiştir:
correlation_values = []
column_list = df.columns.tolist()

for i in column_list:
    correlation = df[" _hum"].corr(df[i])
    correlation_values.append(correlation)
    
plt.figure(figsize=(8, 10))
plt.bar(column_list, correlation_values)
plt.xticks(rotation=90)
plt.xlabel("Columns")
plt.ylabel("Correlation")
plt.title("Korelasyon: _hum ve Diğer Kolonlar Arasında")
plt.tight_layout()
plt.show()

Aykırı Değerlerin İncelenmesi
Sayısal kolonlardaki aykırı değerler aşağıdaki kod kullanılarak incelenmiştir:
T = [" _dewptm", " _hum", " _tempm"," _wspdm"]

for i in T:
    sns.boxplot(x=df[i])
    plt.title(i)
    plt.show()
    q1 = df[i].describe(percentiles=[0.25])["25%"]
    q3 = df[i].describe(percentiles=[0.75])["75%"]
    a = q3 - q1
    lower_bound = q1 - 1.5 * a
    upper_bound = q3 + 1.5 * a
    outliers = df[i][(df[i] < lower_bound) | (df[i] > upper_bound)]
    print(f"{i} için aykırı veri sayısı:",len(outliers))
    print(outliers)
    print("Aykırı verinin yüzdeliği:",100*(len(outliers)/len(df)))
    print("-------------------------------------------------------------")
    print(f"{i} için İstatiksel veriler")
    print(df[i].describe())
Bu kodlar kullanılarak aykırı değerler incelenmiş ve gerektiğinde veri setinden çıkarılmıştır.

Kolon Silme ve Eksik Değerlerin İşlenmesi
"_wspdm" kolonu aykırı değerlerin yüzdesinin yüksek olması nedeniyle aşağıdaki kod kullanılarak kaldırılmıştır:
df.drop(" _wspdm", axis=1, inplace=True)
"_hum" kolonunda eksik değerler, ortalama değer ile doldurulmuştur:
df[" _hum"].fillna(df[" _hum"].mean(), inplace=True)
"_tempm" ve "_dewptm" kolonlarında ise median veya mod kullanılamamıştır çünkü bu yaklaşımlar sapmalara neden olmaktadır.

Veri Temizleme Devamı
Bu bölümde, veri setinden eksik değer içeren satırların ve aykırı değerlerin temizlenmesi ile ilgili Python kodlarına ve sonuçlarına yer verilmiştir.

Eksik Değerleri Silme
" _tempm" ve " _dewptm" kolonlarındaki eksik değerleri içeren satırlar aşağıdaki kod kullanılarak silinmiştir:
df = df.dropna(subset=[" _tempm"])
df = df.dropna(subset=[" _dewptm"])

Aykırı Değerleri Silme
" _dewptm", " _hum", ve " _tempm" kolonlarındaki aykırı değerler, IQR (Interquartile Range) yöntemi kullanılarak tespit edilmiş ve silinmiştir:
column_list = [" _dewptm", " _hum", " _tempm"]

for column in column_list:
    q1 = df[column].describe(percentiles=[0.25])["25%"]
    q3 = df[column].describe(percentiles=[0.75])["75%"]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[column][(df[column] < lower_bound) | (df[column] > upper_bound)]
    df = df.drop(outliers.index)

Aykırı Değerlerin Görselleştirilmesi
Kolonlar arasındaki aykırı değerlerin görselleştirilmesi için aşağıdaki kod kullanılmıştır:

# Kutu grafiği oluşturma
plt.figure(figsize=(8, 6))
X.boxplot()
plt.xticks(rotation=90)
plt.title("Aykırı Değerler: X Kolonları")
plt.show()
Multiple Linear Regression
Bu bölümde, Multiple Linear Regression (Çoklu Doğrusal Regresyon) modeli oluşturulmuş ve değerlendirilmiştir. Bu model, bağımlı değişken " _hum" ile bu değişkenle ilişkilendirilen bağımsız değişkenler " _dewptm" ve " _tempm" arasındaki ilişkiyi analiz eder. İşte bu işlemle ilgili Python kodları:

# Bağımlı ve bağımsız değişkenlerin tanımlanması
column_list = df.columns.tolist()
column_list.remove(" _hum")
X = df[column_list]
y = df[" _hum"]

# Çoklu Doğrusal Regresyon modelinin oluşturulması ve eğitilmesi
model = LinearRegression()
model.fit(X, y)

# Tahmin yapma
y_pred_linear = model.predict(X)

# R-squared (R2) değeri hesaplama
score1 = model.score(X, y)
print(score1)

# Mean Squared Error (MSE) hesaplama
mse1 = mean_squared_error(y, y_pred_linear)
print(mse1)
Bu kodlarla veri seti temizleme, aykırı değerlerin işlenmesi ve regresyon modeli oluşturma ve değerlendirme işlemleri açıklanmıştır.

Hata Ölçümleri: MSE ve RMSE
Bu bölümde, model performansını değerlendirmek için kullanılan hata ölçümleri olan Mean Squared Error (MSE) ve Root Mean Squared Error (RMSE) hesaplamaları ve sonuçları yer almaktadır.

## RMSE (Root Mean Squared Error)
MSE'nin karekökü alınarak elde edilen RMSE, tahminlerin gerçek değerlerden ne kadar sapma gösterdiğini ölçen bir metriktir. İşte bu ölçümün hesaplanması:
# RMSE (Root Mean Squared Error)
rmse1 = np.sqrt(mean_squared_error(y, y_pred_linear))
print("RMSE:", rmse1)
Polinom Regresyon Modeli
Polinom regresyon modeli oluşturulmuş ve bu modelin performansı değerlendirilmiştir. İşte bu işlemle ilgili Python kodları:

# Polinom regresyon modelini oluşturma
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

# Modeli X_poly ve y verilerine uydurma (eğitme)
model.fit(X_poly, y)

# Tahmin yapma
y_pred_poly = model.predict(X_poly)
score2 = model.score(X_poly, y)
print(score2)

mse2 = mean_squared_error(y, y_pred_poly)
print("MSE:", mse2)

Karşılaştırma: Polinom Regresyon vs Lineer Regresyon
Polinom regresyon ve lineer regresyon modellerinin performansı karşılaştırılmıştır. İşte bu karşılaştırmanın görselleştirilmesi:

plt.scatter(y, y_pred_linear, color='purple', label='Lineer Regresyon', s=1)
plt.scatter(y, y_pred_poly, color='lightblue', label='Polinom Regresyon', s=1)
plt.scatter(y, y, color='red', label='Gerçek y', s=1)  # Gerçek y değeri
plt.xlabel('Gerçek y')
plt.ylabel('Tahmin edilen y')
plt.title('Lineer ve Polinom Regresyon Karşılaştırılması')
plt.legend()
plt.show()
Karşılaştırma: MSE ve RMSE
Polinom ve lineer regresyon modellerinin MSE ve RMSE değerleri karşılaştırılmıştır:

# Hata değerlerini bir liste haline getir
mse_values = [mse1, mse2]
rmse_values = [rmse1, rmse2]

# Hata değerlerinin isimlerini bir liste haline getir
labels = ['Linear', 'Polynomial']

# MSE değerlerini görselleştir
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(labels, mse_values, color=['blue', 'red'])
plt.title('Mean Squared Error')
plt.xlabel('Model')
plt.ylabel('Error')

# RMSE değerlerini görselleştir
plt.subplot(1, 2, 2)
plt.bar(labels, rmse_values, color=['blue', 'red'])
plt.title('Root Mean Squared Error')
plt.xlabel('Model')
plt.ylabel('Error')

plt.tight_layout()
plt.show()
Bu kodlarla, regresyon modellerinin performansı ölçülmüş ve karşılaştırılmıştır.

Polinom Regresyon Modeli ile Train-Test Split
Bu bölümde, Polinom Regresyon modeli kullanılarak Train-Test bölünmüş veri seti üzerinde modelin oluşturulması, eğitilmesi ve performansının değerlendirilmesi işlemleri yer almaktadır.
Polinom Özellikleri
Polinom regresyon modeli oluşturulmadan önce, bağımsız değişkenlere polinom özellikleri eklenmiştir:

# Polinom özelliklerini oluşturma
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)
Polinom Regresyon Modeli
Polinom regresyon modeli aşağıdaki kod kullanılarak oluşturulmuş ve eğitilmiştir:

# Polinom regresyon modelini oluşturma
model = LinearRegression()

# Modeli eğitim verilerine uydurma (eğitme)
model.fit(X_train_poly, y_train)

# Test seti üzerinde tahmin yapma
y_pred_poly = model.predict(X_test_poly)

# Modelin test seti üzerindeki başarısını değerlendirme (R-kare değeri)
score4 = model.score(X_test_poly, y_test)
print("R-kare değeri:", score4)

mse4 = mean_squared_error(y_test, y_pred_poly)
print("MSE:", mse4)

# RMSE (Root Mean Squared Error)
rmse4 = np.sqrt(mse4)
print("RMSE:", rmse4)

Karşılaştırma: Polinom Regresyon vs Lineer Regresyon
Polinom regresyon ve lineer regresyon modelleri karşılaştırılmış ve sonuçlar aşağıdaki şekilde görselleştirilmiştir:

# Scatter plot oluşturma
plt.scatter(y_test, y_pred_linear, color='purple', label='Lineer Regresyon', s=1)
plt.scatter(y_test, y_pred_poly, color='lightblue', label='Polinom Regresyon', s=1)
plt.scatter(y_test, y_test, color='red', label='Gerçek y', s=1)  # Gerçek y değeri
plt.xlabel('Gerçek y')
plt.ylabel('Tahmin edilen y')
plt.title('Lineer ve Polinom Regresyon Karşılaştırılması')
plt.legend()
plt.show()

Karşılaştırma: MSE ve RMSE
Polinom ve lineer regresyon modellerinin MSE ve RMSE değerleri karşılaştırılmıştır:

# Hata değerlerini bir liste haline getir
mse_values = [mse3, mse4]
rmse_values = [rmse3, rmse4]

# Hata değerlerinin isimlerini bir liste haline getir
labels = ['Linear', 'Polynomial']

# MSE değerlerini görselleştir
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(labels, mse_values, color=['blue', 'red'])
plt.title('Mean Squared Error')
plt.xlabel('Model')
plt.ylabel('Error')

# RMSE değerlerini görselleştir
plt.subplot(1, 2, 2)
plt.bar(labels, rmse_values, color=['blue', 'red'])
plt.title('Root Mean Squared Error')
plt.xlabel('Model')
plt.ylabel('Error')

plt.tight_layout()
plt.show()
Bu kodlarla, Polinom Regresyon modeli ile Lineer Regresyon modelinin karşılaştırılması ve performans değerlendirmesi yapılmıştır. 

CatBoostRegressor ile Tahminleme
CatBoostRegressor, gradient boosting tabanlı bir regresyon modelidir ve kategorik verilerin bulunduğu veri setlerinde tercih edilir. Aşağıda, CatBoostRegressor modelinin nasıl kullanılacağını ve performansının nasıl değerlendirileceğini gösteren bir örnek bulunmaktadır:

# CatBoost modeli oluşturma
model = CatBoostRegressor()

# Modeli eğitme
model.fit(X_train, y_train, verbose=False)  # verbose=False eğitim sürecinde her iterasyonu göstermeyecektir

# Tahminler oluşturma
y_pred_boost = model.predict(X_test)

# Modelin R2 skorunu hesaplama
score5 = r2_score(y_test, y_pred_boost)

print("R2 Score: ", score5)

Ayrıca, bu modelin MSE (Mean Squared Error) ve RMSE (Root Mean Squared Error) değerlerini hesaplamak için aşağıdaki kodları kullanabilirsiniz:

# MSE hesaplama
mse5 = mean_squared_error(y_test, y_pred_boost)

# RMSE hesaplama
rmse5 = np.sqrt(mse5)

print("Mean Squared Error: ", mse5)
print("Root Mean Squared Error: ", rmse5)
LightGBMRegressor ile Tahminleme
LightGBMRegressor, gradient boosting tabanlı bir regresyon modelidir ve büyük veri setlerinde daha hızlı ve düşük bellek kullanımı ile avantaj sağlar. Aşağıda, LightGBMRegressor modelinin kullanımını ve performans değerlendirmesini gösteren bir örnek bulunmaktadır:

# LightGBMRegressor modeli oluşturma
model = LGBMRegressor()
model.fit(X_train, y_train)
y_pred_lightgbm = model.predict(X_test)

# Modelin R2 skorunu hesaplama
scorelg = r2_score(y_test, y_pred_lightgbm)
print("R2 ", scorelg)

# MSE hesaplama
mselg = mean_squared_error(y_test, y_pred_lightgbm)
rmselg = np.sqrt(mselg)
print("Mean Squared Error: ", mselg)
print("Root Mean Squared Error: ", rmselg)

Tüm Skorların Görselleştirilmesi
Son olarak, farklı modellerin performansını karşılaştırmak için aşağıdaki kodları kullanabilirsiniz:

print(mse1, mse3, mse2, mse4, mse5, mselg)
print(rmse1, rmse3, rmse2, rmse4, rmse5, rmselg)
print(score1, score3, score2, score4, score5, scorelg)

# Hesaplanan hata değerlerini ve skorları bir liste haline getir
mse_values = [mse1, mse3, mse2, mse4, mse5, mselg]
rmse_values = [rmse1, rmse3, rmse2, rmse4, rmse5, rmselg]
score_values = [score1, score3, score2, score4, score5, scorelg]

# Modellerin isimlerini bir liste haline getir
labels = ['Linear', 'Split Linear', 'Polynomial', 'Split Polynomial', 'Catboost', 'LightGBM']

# MSE değerlerini görselleştir
plt.figure(figsize=(20, 6))
plt.subplot(1, 3, 1)
plt.bar(labels, mse_values, color=['blue', 'red', 'green', 'purple', 'yellow', 'pink'])
plt.title('Mean Squared Error')
plt.xlabel('Model')
plt.ylabel('Error')
plt.xticks(rotation=45, ha='right')

# RMSE değerlerini görselleştir
plt.subplot(1, 3, 2)
plt.bar(labels, rmse_values, color=['blue', 'red', 'green', 'purple', 'yellow', 'pink'])
plt.title('Root Mean Squared Error')
plt.xlabel('Model')
plt.ylabel('Error')
plt.xticks(rotation=45, ha='right')

# Score değerlerini görselleştir
plt.subplot(1, 3, 3)
plt.bar(labels, score_values, color=['blue', 'red', 'green', 'purple', 'yellow', 'pink'])
plt.title('Score')
plt.xlabel('Model')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()
Bu kodlarla, farklı regresyon modellerinin performansını karşılaştırabilir ve sonuçları görsel olarak inceleyebilirsiniz.
LightGBM büyük veri setlerinde daha hızlıdır ve daha az bellek kullanır.Bu sebeple daha büyük veri setlerinde LightGBM tercih edilir.
