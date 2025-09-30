import os
import pydicom
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QListWidget, QLabel, QHBoxLayout, QProgressBar,
    QMessageBox, QApplication
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class DICOMLoaderTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent

        self.patients_dict = {}
        self.progress_bar = QProgressBar()
        self.image_canvas = FigureCanvas(Figure(figsize=(5, 5)))
        self.ax = self.image_canvas.figure.subplots()
        self.send_button = QPushButton("📤 İnceleme Sekmesine Gönder")
        self.send_button.clicked.connect(self.send_to_diffusion_tab)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.load_button = QPushButton("📂 DICOM Klasörü Seç")
        self.load_button.clicked.connect(self.load_dicom_folder)
        layout.addWidget(self.load_button)
        layout.addWidget(self.progress_bar)

        list_layout = QHBoxLayout()
        self.patient_list = QListWidget()
        self.patient_list.setMinimumWidth(250)
        self.patient_list.itemClicked.connect(self.update_series_list)
        list_layout.addWidget(self.patient_list)

        self.series_list = QListWidget()
        self.series_list.setMinimumWidth(300)
        list_layout.addWidget(self.series_list)

        layout.addLayout(list_layout)
        layout.addWidget(self.send_button)
        layout.addWidget(QLabel("Görüntü Önizleme:"))
        layout.addWidget(self.image_canvas)
        self.setLayout(layout)

    def load_dicom_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "DICOM klasörü seç")
        if not folder:
            return

        self.patients_dict = {}
        dcm_files = []

        for root, _, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                    dcm_files.append(file_path)
                except:
                    continue

        total_files = len(dcm_files)
        if total_files == 0:
            QMessageBox.warning(self, "DICOM", "Hiç DICOM dosyası bulunamadı!")
            self.progress_bar.setValue(0)
            return

        self.progress_bar.setMaximum(total_files)
        self.progress_bar.setValue(0)
        QApplication.processEvents()

        for i, file_path in enumerate(dcm_files):
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                patient_name = str(ds.get("PatientName", "Unknown"))
                series_desc = str(ds.get("SeriesDescription", "UnnamedSeries"))

                if patient_name not in self.patients_dict:
                    self.patients_dict[patient_name] = {}
                if series_desc not in self.patients_dict[patient_name]:
                    self.patients_dict[patient_name][series_desc] = []
                self.patients_dict[patient_name][series_desc].append(file_path)
            except Exception as e:
                print(f"[HATA] {file_path}: {e}")
            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()

        self.populate_patient_list()
        QMessageBox.information(self, "Yükleme Tamamlandı", f"{total_files} DICOM dosyası başarıyla yüklendi!")

    def populate_patient_list(self):
        self.patient_list.clear()
        for patient in self.patients_dict:
            self.patient_list.addItem(patient)

    def update_series_list(self, item):
        patient = item.text()
        self.series_list.clear()
        if patient in self.patients_dict:
            for series in self.patients_dict[patient]:
                self.series_list.addItem(series)
            self.series_list.itemClicked.connect(
                lambda s_item: self.show_first_image(patient, s_item.text())
            )

    def show_first_image(self, patient, series):
        try:
            file_list = self.patients_dict[patient][series]
            file_list.sort()
            ds = pydicom.dcmread(file_list[0])
            image = ds.pixel_array
            self.ax.clear()
            self.ax.imshow(image, cmap='gray')
            self.ax.set_title(f"{patient} - {series}")
            self.ax.axis('off')
            self.image_canvas.draw()
        except Exception as e:
            print(f"Görüntü gösterilemedi: {e}")

    def send_to_diffusion_tab(self):
        selected_patient = self.patient_list.currentItem()
        selected_series = self.series_list.currentItem()
        if not selected_patient or not selected_series:
            QMessageBox.warning(self, "Uyarı", "Lütfen hasta ve sekans seçiniz.")
            return

        patient = selected_patient.text()
        series = selected_series.text()
        try:
            file_list = self.patients_dict[patient][series]
            file_list.sort()
            if hasattr(self.main_window, 'tab2'):
                self.main_window.tab2.load_diffusion_data(patient, series, file_list)
                QMessageBox.information(self, "Başarılı", "Difüzyon verisi gönderildi.")
            else:
                QMessageBox.warning(self, "Hata", "Difüzyon sekmesine ulaşılamadı.")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Veri gönderilemedi: {e}")
