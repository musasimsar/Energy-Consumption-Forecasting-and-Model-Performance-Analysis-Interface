import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PyQt5.QtWidgets import QApplication
from gui import MainWindow
from ml_models import MLModels
from dl_models import DLModels

# Veri Yükleme ve Hazırlama
file_path = "household_power_consumption.csv"
data = pd.read_csv(file_path, sep=",", na_values=['?'], low_memory=False)

# Tarih ve zaman sütunlarını birleştirip yeni bir DateTime sütunu oluştur
data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')

# Yıl, ay, gün gibi sayısal sütunlar ekle
data['year'] = data['DateTime'].dt.year
data['month'] = data['DateTime'].dt.month
data['day'] = data['DateTime'].dt.day
data['hour'] = data['DateTime'].dt.hour
data['minute'] = data['DateTime'].dt.minute

# Tarih ve zaman sütunlarını kaldır
data = data.drop(columns=['Date', 'Time', 'DateTime'])

# Global_active_power'ı float'a çevir
data['Global_active_power'] = data['Global_active_power'].astype(float)

# Veri ön işleme fonksiyonu
def preprocess_data(data, target_column, sample_size=100, random_state=42):
    # 1. Hedef değişkeni ve X veri setini ayır
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # 2. NaN içeren satırları X ve y'den kaldır
    X = X.dropna()
    y = y[X.index]  # X ile y'yi senkronize et
    
    # 3. Belirtilen sayıda rastgele örnekle
    X = X.sample(n=sample_size, random_state=random_state)
    y = y[X.index]  # Seçilen örneklemle y'yi eşleştir
    
    return X, y

# Veri temizleme ve işleme
X, y = preprocess_data(data, 'Global_active_power')

# Veriyi normalleştirme
scaler = StandardScaler()
X = scaler.fit_transform(X)

# X ve y'yi numpy dizilerine çevirme
X = np.array(X)
y = np.array(y)

# X ve y'yi train/test olarak bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PyQt5 uygulamasını başlatma
def run_app():
    app = QApplication(sys.argv)
    window = MainWindow(X_train, X_test, y_train, y_test, data)  # Veriyi GUI'ye geçiriyoruz
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_app()
