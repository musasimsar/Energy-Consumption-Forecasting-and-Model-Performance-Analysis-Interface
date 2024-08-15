import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit, QFormLayout, QSpinBox, QDialog, QDialogButtonBox, QTableWidget, QTableWidgetItem, QLabel, QComboBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from sklearn.model_selection import train_test_split
from ml_models import MLModels
from dl_models import DLModels
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt5.QtGui import QImage, QPixmap

class MainWindow(QMainWindow):
    def __init__(self, X_train, X_test, y_train, y_test, data):
        super().__init__()

        self.setWindowTitle("Model Performans Analizi")
        self.setGeometry(100, 100, 1200, 800)

        # Veriyi ve modül nesnelerini sakla
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.data = data
        self.ml_models = MLModels(X_train, y_train, X_test, y_test)
        self.dl_models = DLModels(X_train, y_train, X_test, y_test)
        self.current_data = None  # İlk 10 satır için değişken

        self.init_ui()

    def init_ui(self):
        # Ana widget ve layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)  # Ana layout dikey olacak şekilde ayarlandı

        # Seçim ekranı widget'ı
        self.selection_widget = QWidget()
        self.selection_layout = QVBoxLayout(self.selection_widget)

        # Üstte bir başlık ekliyoruz
        self.title_label = QLabel("İşlem Seçiniz")
        self.title_label.setFont(QFont('Arial', 16))
        self.title_label.setAlignment(Qt.AlignCenter)

        # İşlem seçimi için combobox ekliyoruz
        self.combobox = QComboBox()
        self.combobox.addItem("İlk 10 Satırı Göster")
        self.combobox.addItem("Decision Tree Çalıştır")
        self.combobox.addItem("Random Forest Çalıştır")
        self.combobox.addItem("ANN Çalıştır")
        self.combobox.addItem("LSTM Çalıştır")

        # Seçimi onaylamak için bir buton ekliyoruz
        self.select_button = QPushButton("Seç")
        self.select_button.clicked.connect(self.select_action)

        # Combobox ve butonu ortalamak için bir layout daha ekliyoruz
        self.selection_layout.addWidget(self.title_label)
        self.selection_layout.addWidget(self.combobox)
        self.selection_layout.addWidget(self.select_button)
        self.selection_layout.setAlignment(Qt.AlignCenter)

        # Ana layout'a seçim layout'unu ekliyoruz
        self.main_layout.addWidget(self.selection_widget)

        # Sonuç ekranı widget'ları
        self.result_widget = QWidget()
        self.result_layout = QVBoxLayout(self.result_widget)

        self.result_area = QTextEdit()
        self.result_area.setReadOnly(True)
        self.result_layout.addWidget(self.result_area)

        self.table_widget = QTableWidget()
        self.table_widget.setVisible(False)
        self.result_layout.addWidget(self.table_widget)

        self.confusion_matrix_label = QLabel()
        self.result_layout.addWidget(self.confusion_matrix_label)

        self.back_button = QPushButton("Geri")
        self.back_button.clicked.connect(self.show_selection_screen)
        self.result_layout.addWidget(self.back_button)

        self.main_layout.addWidget(self.result_widget)
        self.result_widget.setVisible(False)

    def show_selection_screen(self):
        self.result_widget.setVisible(False)
        self.selection_widget.setVisible(True)

    def show_result_screen(self):
        self.selection_widget.setVisible(False)
        self.result_widget.setVisible(True)

    def select_action(self):
        action = self.combobox.currentText()
        if action == "İlk 10 Satırı Göster":
            self.show_first_ten_rows()
        elif action == "Decision Tree Çalıştır":
            self.run_decision_tree()
        elif action == "Random Forest Çalıştır":
            self.run_random_forest()
        elif action == "ANN Çalıştır":
            self.run_ann()
        elif action == "LSTM Çalıştır":
            self.run_lstm()

    def show_first_ten_rows(self):
        self.clear_result_area()
        self.table_widget.setRowCount(10)
        self.table_widget.setColumnCount(self.data.shape[1])
        self.table_widget.setHorizontalHeaderLabels(self.data.columns)

        for row in range(10):
            for col in range(self.data.shape[1]):
                self.table_widget.setItem(row, col, QTableWidgetItem(str(self.data.iloc[row, col])))

        self.table_widget.setVisible(True)
        self.result_area.setVisible(False)
        self.show_result_screen()

    def run_decision_tree(self):
        self.clear_result_area()
        test_size, ok = self.get_test_size()
        if ok:
            X_train, X_test, y_train, y_test = train_test_split(self.X_train, self.y_train, test_size=test_size, random_state=42)
            self.ml_models = MLModels(X_train, y_train, X_test, y_test)
            mse, r2 = self.ml_models.decision_tree()
            self.result_area.setVisible(True)
            self.result_area.append(f'Decision Tree Regressor\nMSE: {mse}, R2: {r2 * 100:.2f}%')
            self.plot_confusion_matrix(self.ml_models.y_test, self.ml_models.models['decision_tree'].predict(self.ml_models.X_test))
            self.show_result_screen()

    def run_random_forest(self):
        self.clear_result_area()
        test_size, ok = self.get_test_size()
        if ok:
            X_train, X_test, y_train, y_test = train_test_split(self.X_train, self.y_train, test_size=test_size, random_state=42)
            self.ml_models = MLModels(X_train, y_train, X_test, y_test)
            mse, r2 = self.ml_models.random_forest()
            self.result_area.setVisible(True)
            self.result_area.append(f'Random Forest Regressor\nMSE: {mse}, R2: {r2 * 100:.2f}%')
            self.plot_confusion_matrix(self.ml_models.y_test, self.ml_models.models['random_forest'].predict(self.ml_models.X_test))
            self.show_result_screen()

    def run_ann(self):
        self.clear_result_area()
        epochs, batch_size, ok = self.get_dl_params()
        if ok:
            mse, r2 = self.dl_models.ann(epochs=epochs, batch_size=batch_size)
            self.result_area.setVisible(True)
            self.result_area.append(f'ANN\nMSE: {mse}, R2: {r2 * 100:.2f}%')
            self.plot_confusion_matrix(self.dl_models.y_test, self.dl_models.ann_model.predict(self.dl_models.X_test))
            self.show_result_screen()

    def run_lstm(self):
        self.clear_result_area()
        epochs, batch_size, ok = self.get_dl_params()
        if ok:
            mse, r2 = self.dl_models.lstm(epochs=epochs, batch_size=batch_size)
            self.result_area.setVisible(True)
            self.result_area.append(f'LSTM\nMSE: {mse}, R2: {r2 * 100:.2f}%')
            self.plot_confusion_matrix(self.dl_models.y_test, self.dl_models.lstm_model.predict(self.dl_models.X_test))
            self.show_result_screen()

    def clear_result_area(self):
        self.result_area.clear()
        self.result_area.setVisible(False)
        self.confusion_matrix_label.clear()
        self.table_widget.setVisible(False)

    def get_test_size(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Test Size Seç")
        dialog_layout = QFormLayout(dialog)
        
        test_size_spinbox = QSpinBox(dialog)
        test_size_spinbox.setRange(10, 50)
        test_size_spinbox.setSuffix('%')
        test_size_spinbox.setValue(20)
        dialog_layout.addRow("Test Size:", test_size_spinbox)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        dialog_layout.addWidget(buttons)
        
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            return test_size_spinbox.value() / 100, True
        return None, False
    
    def get_dl_params(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Epoch ve Batch Size Seç")
        dialog_layout = QFormLayout(dialog)
        
        epoch_spinbox = QSpinBox(dialog)
        epoch_spinbox.setRange(1, 200)
        epoch_spinbox.setValue(50)
        dialog_layout.addRow("Epoch:", epoch_spinbox)
        
        batch_size_spinbox = QSpinBox(dialog)
        batch_size_spinbox.setRange(1, 256)
        batch_size_spinbox.setValue(32)
        dialog_layout.addRow("Batch Size:", batch_size_spinbox)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        dialog_layout.addWidget(buttons)
        
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            return epoch_spinbox.value(), batch_size_spinbox.value(), True
        return None, None, False
    
    def plot_confusion_matrix(self, y_true, y_pred):
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(pd.DataFrame({'Actual': y_true.flatten(), 'Predicted': y_pred.flatten()}).round(2).corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        ax.set_title("Confusion Matrix")
        
        # figi QPixmap olarak alıp QLabel üzerinde gösterelim
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
        plt.close(fig)

        qimg = QImage(img.data, width, height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.confusion_matrix_label.setPixmap(pixmap)
