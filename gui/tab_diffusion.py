import numpy as np
import pydicom
from collections import defaultdict
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSlider, QHBoxLayout,
    QComboBox, QPushButton,QLineEdit,QCheckBox,QDialog
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import EllipseSelector, LassoSelector, PolygonSelector
from matplotlib.path import Path
from scipy.optimize import curve_fit
import pymc as pm

import pandas as pd
from datetime import datetime
import os

def ivim_bi_model(b, f, D, D_star):
    return f * np.exp(-b * D_star) + (1 - f) * np.exp(-b * D)
def ivim_bis_model(b, f, D_star, D_fixed):
    return f * np.exp(-b * D_star) + (1 - f) * np.exp(-b * D_fixed)
class PatientInfoDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hasta Bilgileri")

        layout = QVBoxLayout()

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Ad")
        layout.addWidget(QLabel("Hasta Adı:"))
        layout.addWidget(self.name_input)

        self.surname_input = QLineEdit()
        self.surname_input.setPlaceholderText("Soyad")
        layout.addWidget(QLabel("Hasta Soyadı:"))
        layout.addWidget(self.surname_input)

        self.save_button = QPushButton("Kaydet ve Aktar")
        self.save_button.clicked.connect(self.accept)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def get_patient_info(self):
        return self.name_input.text(), self.surname_input.text()

class DiffusionViewerTab(QWidget):
    def __init__(self):
        super().__init__()

        self.image_data = {}
        self.b_values = []
        self.slice_positions = []
        self.current_b_index = 0
        self.current_slice_index = 0

        self.roi_selector = None
        self.roi_mask = None

        self.b_label = QLabel("b-değeri: -")
        self.slice_label = QLabel("Slice: -")

        self.b_slider = QSlider(Qt.Horizontal)
        self.b_slider.valueChanged.connect(self.update_b_value)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self.update_image)

        self.roi_combo = QComboBox()
        self.roi_combo.addItems(["Daire (Ellipse)", "Serbest Elle", "Poligon"])

        self.roi_button = QPushButton("ROI Çiz")
        self.roi_button.clicked.connect(self.start_roi)

        self.image_canvas = FigureCanvas(Figure(figsize=(8, 8)))
        self.ax = self.image_canvas.figure.subplots()

        self.graph_canvas = FigureCanvas(Figure(figsize=(5, 5)))
        self.graph_ax = self.graph_canvas.figure.subplots()

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()

        left_panel = QVBoxLayout()
        b_layout = QHBoxLayout()
        b_layout.addWidget(self.b_label)
        b_layout.addWidget(self.b_slider)
        left_panel.addLayout(b_layout)

        s_layout = QHBoxLayout()
        s_layout.addWidget(self.slice_label)
        s_layout.addWidget(self.slice_slider)
        left_panel.addLayout(s_layout)

        roi_layout = QHBoxLayout()
        roi_layout.addWidget(QLabel("ROI Tipi:"))
        roi_layout.addWidget(self.roi_combo)
        roi_layout.addWidget(self.roi_button)
        left_panel.addLayout(roi_layout)

        left_panel.addWidget(self.image_canvas)

        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("Sinyal Azalımı Grafiği"))
        right_panel.addWidget(self.graph_canvas)

        main_layout.addLayout(left_panel, stretch=3)
        main_layout.addLayout(right_panel, stretch=2)
        self.adc_button = QPushButton("ADC Hesapla")
        self.adc_button.clicked.connect(self.compute_adc)

        self.ivim_button = QPushButton("IVIM (Basit)")
        self.ivim_button.clicked.connect(self.compute_ivim_bi)  # Bir sonraki adımda eklenecek

        self.exclude_b0_checkbox = QCheckBox("b=0’ı hariç tut")
        self.exclude_b0_checkbox.setChecked(False)  # Varsayılan olarak dahil
        left_panel.addWidget(self.exclude_b0_checkbox)


        self.map_type_combo = QComboBox()
        self.map_type_combo.addItems([
    "ADC", "IVIM_f", "IVIM_D", "IVIM_D*",
    "IVIM_SEG_f", "IVIM_SEG_D", "IVIM_SEG_D*"
])

        self.map_button = QPushButton("Harita Oluştur")
        self.map_button.clicked.connect(self.create_param_map_by_type)     # Sonraki adım
        self.ivim_seg_button = QPushButton("IVIM (Segmentli)")
        self.ivim_seg_button.clicked.connect(self.compute_ivim_segmented)
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Min:"))
        self.vmin_input = QLineEdit()
        self.vmin_input.setFixedWidth(60)
        scale_layout.addWidget(self.vmin_input)

        scale_layout.addWidget(QLabel("Max:"))
        self.vmax_input = QLineEdit()
        self.vmax_input.setFixedWidth(60)
        scale_layout.addWidget(self.vmax_input)

        self.apply_scale_button = QPushButton("Skalayı Uygula")
        self.apply_scale_button.clicked.connect(self.apply_colormap_scale)
        scale_layout.addWidget(self.apply_scale_button)

        left_panel.addLayout(scale_layout)
        self.ivim_bayes_button = QPushButton("IVIM (Bayesian)")
        self.ivim_bayes_button.clicked.connect(self.compute_ivim_bayesian)

        self.clear_roi_button = QPushButton("ROI’yi Temizle")
        self.clear_roi_button.clicked.connect(self.clear_roi)

        left_panel.addWidget(self.clear_roi_button)

        fit_layout = QHBoxLayout()
        fit_layout.addWidget(self.adc_button)
        fit_layout.addWidget(self.ivim_button)
        fit_layout.addWidget(self.map_button)
        fit_layout.addWidget(self.ivim_seg_button)
        fit_layout.addWidget(self.ivim_bayes_button)

        left_panel.addLayout(fit_layout)
        map_layout = QHBoxLayout()
        map_layout.addWidget(QLabel("Harita Tipi:"))
        map_layout.addWidget(self.map_type_combo)
        map_layout.addWidget(self.map_button)
        self.export_button = QPushButton("Sonuçları Excel'e Aktar")
        self.export_button.clicked.connect(self.export_results_to_excel)
        left_panel.addWidget(self.export_button)
        left_panel.addLayout(map_layout)
        self.setLayout(main_layout)

    def load_diffusion_data(self, patient, series, file_list):
        self.image_data.clear()
        self.b_values.clear()
        self.slice_positions.clear()

        b_slice_dict = defaultdict(dict)

        for file in file_list:
            try:
                ds = pydicom.dcmread(file)
                b_val = int(ds[(0x0019, 0x100C)].value)
                slice_pos = tuple(ds.ImagePositionPatient)
                img = ds.pixel_array
                b_slice_dict[b_val][slice_pos] = img
            except Exception as e:
                print(f"[HATA] {file}: {e}")

        self.b_values = sorted(b_slice_dict.keys())
        self.slice_positions = sorted({pos for d in b_slice_dict.values() for pos in d})

        for b_val in self.b_values:
            self.image_data[b_val] = []
            for pos in self.slice_positions:
                if pos in b_slice_dict[b_val]:
                    self.image_data[b_val].append(b_slice_dict[b_val][pos])
                else:
                    ref_shape = next(iter(b_slice_dict[b_val].values())).shape
                    self.image_data[b_val].append(np.zeros(ref_shape, dtype=np.uint16))

        self.b_slider.setMaximum(len(self.b_values) - 1)
        self.b_slider.setValue(0)
        self.slice_slider.setMaximum(len(self.slice_positions) - 1)
        self.slice_slider.setValue(0)
        self.current_b_index = 0
        self.current_slice_index = 0

        self.update_image()

    def update_b_value(self):
        self.current_b_index = self.b_slider.value()
        b_val = self.b_values[self.current_b_index]
        self.b_label.setText(f"b-değeri: {b_val}")
        self.slice_slider.setValue(self.current_slice_index)
        self.update_image()

    def update_image(self):
        b_val = self.b_values[self.current_b_index]
        self.current_slice_index = self.slice_slider.value()
        self.slice_label.setText(f"Slice: {self.current_slice_index}")
        img = self.image_data[b_val][self.current_slice_index]

        self.ax.clear()
        self.ax.imshow(img, cmap='gray')
        self.ax.set_title(f"b={b_val}, slice={self.current_slice_index}")
        self.ax.axis('off')
        self.image_canvas.draw()

    def start_roi(self):
        img = self.image_data[self.b_values[self.current_b_index]][self.current_slice_index]
        ny, nx = img.shape

        self.ax.clear()
        self.ax.imshow(img, cmap='gray')
        self.ax.set_title("ROI çiz: tamamladığında bırak")
        self.ax.axis('off')

        if self.roi_selector:
            self.roi_selector.set_visible(False)
            self.roi_selector.disconnect_events()
            self.roi_selector = None

        roi_type = self.roi_combo.currentText()
        if roi_type == "Daire (Ellipse)":
            self.roi_selector = EllipseSelector(
                self.ax,
                onselect=self.roi_done_ellipse,
                interactive=True,
                useblit=True,
                button=[1]
            )
        elif roi_type == "Serbest Elle":
            self.roi_selector = LassoSelector(
                self.ax,
                onselect=self.roi_done_freehand
            )
        elif roi_type == "Poligon":
            self.roi_selector = PolygonSelector(
                self.ax,
                onselect=self.roi_done_polygon,
                useblit=True
            )
       

        self.image_canvas.draw()

    def roi_done_ellipse(self, eclick, erelease):
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata

        img = self.image_data[self.b_values[self.current_b_index]][self.current_slice_index]
        ny, nx = img.shape
        rr, cc = np.ogrid[:ny, :nx]

        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = abs(x1 - x0) / 2, abs(y1 - y0) / 2

        mask = ((cc - cx)**2 / rx**2 + (rr - cy)**2 / ry**2) <= 1
        if self.roi_mask is None:
            self.roi_mask = mask
        else:
            self.roi_mask |= mask

        print("ROI (ellipse) tamamlandı")
        self.map_f = None
        self.map_D = None
        self.map_D_star = None
        self.compute_signal_plot()
        # ROI birleşik maskesini overlay olarak göster
        self.ax.clear()
        b_vals = np.array(self.b_values)
        b_fit = b_vals[b_vals <= 1000]
        base_img = self.image_data[int(b_fit[0])][self.current_slice_index]
        self.ax.imshow(base_img, cmap='gray')
        self.ax.imshow(self.roi_mask, cmap='autumn', alpha=0.3)
        self.ax.set_title("Çoklu ROI Görselleştirme")
        self.ax.axis('off')
        self.image_canvas.draw()


    def roi_done_freehand(self, verts):
        img = self.image_data[self.b_values[self.current_b_index]][self.current_slice_index]
        ny, nx = img.shape
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        points = np.vstack((x.flatten(), y.flatten())).T
        path = Path(verts)
        mask = path.contains_points(points).reshape((ny, nx))
        if self.roi_mask is None:
            self.roi_mask = mask
        else:
            self.roi_mask |= mask

        print("ROI (lasso) tamamlandı")
        self.map_f = None
        self.map_D = None
        self.map_D_star = None
        self.compute_signal_plot()
        # ROI birleşik maskesini overlay olarak göster
        self.ax.clear()
        b_vals = np.array(self.b_values)
        b_fit = b_vals[b_vals <= 1000]
        base_img = self.image_data[int(b_fit[0])][self.current_slice_index]
        self.ax.imshow(base_img, cmap='gray')
        self.ax.imshow(self.roi_mask, cmap='autumn', alpha=0.3)
        self.ax.set_title("Çoklu ROI Görselleştirme")
        self.ax.axis('off')
        self.image_canvas.draw()


    def roi_done_polygon(self, verts):
        img = self.image_data[self.b_values[self.current_b_index]][self.current_slice_index]
        ny, nx = img.shape
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        points = np.vstack((x.flatten(), y.flatten())).T
        path = Path(verts)
        mask = path.contains_points(points).reshape((ny, nx))
        if self.roi_mask is None:
            self.roi_mask = mask
        else:
            self.roi_mask |= mask

        print("ROI (polygon) tamamlandı")
        self.map_f = None
        self.map_D = None
        self.map_D_star = None
        self.compute_signal_plot()
        # ROI birleşik maskesini overlay olarak göster
        self.ax.clear()
        b_vals = np.array(self.b_values)
        b_fit = b_vals[b_vals <= 1000]
        base_img = self.image_data[int(b_fit[0])][self.current_slice_index]
        self.ax.imshow(base_img, cmap='gray')
        self.ax.imshow(self.roi_mask, cmap='autumn', alpha=0.3)
        self.ax.set_title("Çoklu ROI Görselleştirme")
        self.ax.axis('off')
        self.image_canvas.draw()


    def compute_signal_plot(self):
        if self.roi_mask is None:
            return

        means = []
        for b_val in self.b_values:
            img = self.image_data[b_val][self.current_slice_index]
            masked_pixels = img[self.roi_mask]
            means.append(np.mean(masked_pixels))

        self.graph_ax.clear()
        self.graph_ax.plot(self.b_values, means, marker='o')
        self.graph_ax.set_xlabel("b-değeri")
        self.graph_ax.set_ylabel("Ortalama Sinyal")
        self.graph_ax.set_title("Sinyal Azalımı")
        self.graph_ax.grid(True)
        self.graph_canvas.draw()
        
    def compute_adc(self):
        if self.roi_mask is None:
            print("ROI yok")
            return

        b_vals = np.array(self.b_values)
        if self.exclude_b0_checkbox.isChecked():
            b_vals = b_vals[b_vals > 0]  # b=0 çık
        mask = b_vals <= 1000
        b_fit = b_vals[mask]

        signals = []
        for b_val in b_fit:
            img = self.image_data[b_val][self.current_slice_index]
            roi_signal = img[self.roi_mask]
            signals.append(np.mean(roi_signal))

        signals = np.array(signals)
        if signals[0] == 0:
            print("S0 = 0, normalize edilemez")
            return

    # Normalize: S(b)/S0
        signals_norm = signals / signals[0]

    # Fit: log(S/S0) = -b * ADC
        y = np.log(signals_norm + 1e-8)  # küçük sabit ile log(0) engellenir
        X = -b_fit.reshape(-1, 1)

    # Lineer fit (ADC = -slope)
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        adc = coeffs[0]

    # Tahmini eğri
        b_plot = np.linspace(0, max(b_fit), 100)
        fit_curve = np.exp(-adc * b_plot)
        self.adc = adc
    # Grafiğe ekle
        self.graph_ax.clear()
        self.graph_ax.plot(b_fit, signals_norm, 'o', label='Normalize Gerçek Sinyal')
        self.graph_ax.plot(b_plot, fit_curve, '--r', label=f'ADC Fit (ADC={adc:.4f} mm²/s)')
        self.graph_ax.set_xlabel("b-değeri")
        self.graph_ax.set_ylabel("Sinyal / S0")
        self.graph_ax.set_title("ADC Fit (Normalize)")
        self.graph_ax.grid(True)
        self.graph_ax.set_ylim(0, 1.05)  # opsiyonel: daha iyi görünüm
        self.graph_ax.legend()
        self.graph_canvas.draw()

    
    
    def compute_ivim_bi(self):
        if self.roi_mask is None:
            print("ROI çizilmemiş.")
            return

        b_vals = np.array(self.b_values)
        if self.exclude_b0_checkbox.isChecked():
            b_vals = b_vals[b_vals > 0]  # b=0 çık
        mask = b_vals <= 1000
        b_fit = b_vals[mask]

        signals = []
        for b_val in b_fit:
            img = self.image_data[b_val][self.current_slice_index]
            roi_signal = img[self.roi_mask]
            signals.append(np.mean(roi_signal))

        signals = np.array(signals)
        if signals[0] == 0:
            print("S0 = 0, normalize edilemez")
            return

        signals_norm = signals / signals[0]

        try:
        # Başlangıç parametreleri: f, D, D*
            p0 = [0.1, 0.001, 0.01]
            bounds = ([0, 1e-5, 1e-3], [1, 0.01, 0.1])  # [min], [max]
            popt, _ = curve_fit(ivim_bi_model, b_fit, signals_norm, p0=p0, bounds=bounds)
            f, D, D_star = popt

            self.free_f,self.free_D,self.free_D_star = f,D,D_star

        # Fit eğrisi oluştur
            b_plot = np.linspace(0, max(b_fit), 100)
            fit_curve = ivim_bi_model(b_plot, *popt)

        # Grafiği çiz
            self.graph_ax.clear()
            self.graph_ax.plot(b_fit, signals_norm, 'o', label='Normalize ROI Sinyali')
            self.graph_ax.plot(b_plot, fit_curve, '--r',
                           label=f'IVIM Fit\nf={f:.3f}, D={D:.4e}, D*={D_star:.4e}')
            self.graph_ax.set_xlabel("b-değeri")
            self.graph_ax.set_ylabel("Sinyal / S₀")
            self.graph_ax.set_title("IVIM Fit (Biexponential)")
            self.graph_ax.set_ylim(0, 1.05)
            self.graph_ax.grid(True)
            self.graph_ax.legend()
            self.graph_canvas.draw()

        except Exception as e:
            print(f"IVIM BI fitting hatası: {e}")

    def compute_ivim_segmented(self):
        if self.roi_mask is None:
            print("ROI çizilmemiş.")
            return

        b_vals = np.array(self.b_values)
        if self.exclude_b0_checkbox.isChecked():
            b_vals = b_vals[b_vals > 0]  # b=0 çık
        mask = b_vals <= 1000
        b_fit = b_vals[mask]

        signals = []
        for b_val in b_fit:
            img = self.image_data[b_val][self.current_slice_index]
            roi_signal = img[self.roi_mask]
            signals.append(np.mean(roi_signal))

        signals = np.array(signals)
        if signals[0] == 0:
            print("S0 = 0, normalize edilemez")
            return

        signals_norm = signals / signals[0]

    # AŞAMA 1: D'yi b > 200 ile tahmin et
        b_high_mask = b_fit > 200
        if np.sum(b_high_mask) < 2:
            print("Yüksek b-değeri noktası çok az. Segmentli fit yapılamaz.")
            return

        X_high = -b_fit[b_high_mask].reshape(-1, 1)
        y_high = np.log(signals_norm[b_high_mask] + 1e-8)
        D_est = np.linalg.lstsq(X_high, y_high, rcond=None)[0][0]

    # AŞAMA 2: f ve D* tahmini, D sabit
        def fit_func(b, f, D_star):
            return ivim_bis_model(b, f, D_star, D_est)
        try:
            p0 = [0.1, 0.01]
            bounds = ([0, 1e-3], [1, 0.1])
            popt, _ = curve_fit(fit_func, b_fit, signals_norm, p0=p0, bounds=bounds)
            f_est, D_star_est = popt
            self.seg_f,self.seg_D,self.seg_D_star = f_est,D_est,D_star_est
        # Fit eğrisi
            b_plot = np.linspace(0, max(b_fit), 100)
            fit_curve = fit_func(b_plot, *popt)

            self.graph_ax.clear()
            self.graph_ax.plot(b_fit, signals_norm, 'o', label='Normalize ROI Sinyali')
            self.graph_ax.plot(b_plot, fit_curve, '--g',
                           label=f'Segmentli IVIM Fit\nf={f_est:.3f}, D={D_est:.4e}, D*={D_star_est:.4e}')
            self.graph_ax.set_xlabel("b-değeri")
            self.graph_ax.set_ylabel("Sinyal / S₀")
            self.graph_ax.set_title("IVIM Fit (Segmentli)")
            self.graph_ax.set_ylim(0, 1.05)
            self.graph_ax.grid(True)
            self.graph_ax.legend()
            self.graph_canvas.draw()

        except Exception as e:
            print(f"Segmentli IVIM fit hatası: {e}")

    def compute_ivim_bayesian(self):
    

        if self.roi_mask is None:
            print("ROI çizilmemiş.")
            return

        b_vals = np.array(self.b_values)
        if self.exclude_b0_checkbox.isChecked():
            b_vals = b_vals[b_vals > 0]  # b=0 çık
        mask = b_vals <= 1000
        b_fit = b_vals[mask]

        signals = []
        for b_val in b_fit:
            img = self.image_data[int(b_val)][self.current_slice_index]
            roi_signal = img[self.roi_mask]
            signals.append(np.mean(roi_signal))

        signals = np.array(signals)
        if signals[0] <= 0 or np.any(signals <= 0):
            print("Geçersiz sinyal değerleri.")
            return

        signals_norm = signals / signals[0]

        try:
            with pm.Model() as ivim_model:
                # Priors
                f = pm.Uniform("f", 0.0, 0.3)
                D = pm.Uniform("D", 1e-4, 0.003)
                D_star = pm.Uniform("D_star", 0.005, 0.1)

                # Model
                S_pred = f * pm.math.exp(-b_fit * D_star) + (1 - f) * pm.math.exp(-b_fit * D)

                # Noise model
                sigma = pm.HalfNormal("sigma", sigma=0.05)
                Y_obs = pm.Normal("Y_obs", mu=S_pred, sigma=sigma, observed=signals_norm)

                # Sampling
                trace = pm.sample(draws=1000, tune=1000, chains=5, target_accept=0.95, progressbar=True)

            # Posterior ortalamaları
            f_est = trace.posterior["f"].mean().item()
            D_est = trace.posterior["D"].mean().item()
            D_star_est = trace.posterior["D_star"].mean().item()

            # Fit eğrisi
            b_plot = np.linspace(0, max(b_fit), 100)
            fit_curve = f_est * np.exp(-b_plot * D_star_est) + (1 - f_est) * np.exp(-b_plot * D_est)
            self.bayes_f,self.bayes_D,self.bayes_D_star = f_est,D_est,D_star_est
            # Grafiği çiz
            self.graph_ax.clear()
            self.graph_ax.plot(b_fit, signals_norm, 'o', label='Normalize ROI Sinyali')
            self.graph_ax.plot(b_plot, fit_curve, '--m',
                            label=f'Bayesian MCMC\nf={f_est:.3f}, D={D_est:.4e}, D*={D_star_est:.4e}')
            self.graph_ax.set_xlabel("b-değeri")
            self.graph_ax.set_ylabel("Sinyal / S₀")
            self.graph_ax.set_title("Bayesian IVIM Fit (MCMC)")
            self.graph_ax.set_ylim(0, 1.05)
            self.graph_ax.grid(True)
            self.graph_ax.legend()
            self.graph_canvas.draw()

        except Exception as e:
            print(f"Bayesian MCMC fitting hatası: {e}")

    def create_param_map(self):
        if self.roi_mask is None:
            print("ROI çizilmemiş.")
            return

        b_vals = np.array(self.b_values)
        if self.exclude_b0_checkbox.isChecked():
            b_vals = b_vals[b_vals > 0]  # b=0 çık
        mask = b_vals <= 1000
        b_fit = b_vals[mask]
        if len(b_fit) < 2:
            print("Yeterli b-değeri yok.")
            return

        img_shape = self.image_data[b_fit[0]][self.current_slice_index].shape
        param_map = np.zeros(img_shape, dtype=np.float32)

        for y in range(img_shape[0]):
            for x in range(img_shape[1]):
                if not self.roi_mask[y, x]:
                    continue

                signal_pixel = []
                for b in b_fit:
                    img = self.image_data[b][self.current_slice_index]
                    signal_pixel.append(img[y, x])
                signal_pixel = np.array(signal_pixel)

                if signal_pixel[0] == 0:
                    continue

                signal_norm = signal_pixel / signal_pixel[0]
                y_log = np.log(signal_norm + 1e-8)
                X = -b_fit.reshape(-1, 1)

                try:
                    coeff = np.linalg.lstsq(X, y_log, rcond=None)[0][0]
                    param_map[y, x] = coeff
                except:
                    param_map[y, x] = 0

    # Görselleştir
        self.ax.clear()
        base_img = self.image_data[b_fit[0]][self.current_slice_index]
        self.ax.imshow(base_img, cmap='gray')
        overlay = self.ax.imshow(param_map, cmap='jet', alpha=0.5)
        self.ax.set_title("ADC Haritası (ROI içi)")
        self.ax.axis('off')
        self.image_canvas.draw()

    def create_param_map_bi(self):
    
        if self.roi_mask is None:
            print("ROI çizilmemiş.")
            return

        from scipy.optimize import curve_fit

        def ivim_model(b, f, D, D_star):
            return f * np.exp(-b * D_star) + (1 - f) * np.exp(-b * D)

    # b-değerleri filtrele (b ≤ 1000)
        b_vals = np.array(self.b_values)
        if self.exclude_b0_checkbox.isChecked():
            b_vals = b_vals[b_vals > 0]  # b=0 çık
        mask = b_vals <= 1000
        b_fit = b_vals[mask]

        if len(b_fit) < 3:
            print("Yeterli b-değeri yok.")
            return

        img_shape = self.image_data[int(b_fit[0])][self.current_slice_index].shape
        map_f = np.zeros(img_shape, dtype=np.float32)
        map_D = np.zeros(img_shape, dtype=np.float32)
        map_D_star = np.zeros(img_shape, dtype=np.float32)

    # ROI içindeki her piksel için fitting
        for y in range(img_shape[0]):
            for x in range(img_shape[1]):
                if not self.roi_mask[y, x]:
                    continue

            # Piksel bazlı sinyali topla
                signal_pixel = []
                for b in b_fit:
                    img = self.image_data[int(b)][self.current_slice_index]
                    signal_pixel.append(img[y, x])
                signal_pixel = np.array(signal_pixel)

            # S0 = 0 veya negatif sinyalleri atla
                if signal_pixel[0] <= 0 or np.any(signal_pixel <= 0):
                    continue

            # Normalize et: S(b) / S0
                signal_norm = signal_pixel / signal_pixel[0]

                try:
                # Fitting parametreleri ve sınırlar (BI yöntemi)
                    p0 = [0.1, 0.001, 0.01]  # [f, D, D*]
                    bounds = ([0, 1e-5, 1e-3], [1, 0.01, 0.1])
                    popt, _ = curve_fit(ivim_model, b_fit, signal_norm, p0=p0, bounds=bounds)
                    f_val, D_val, D_star_val = popt
                    map_f[y, x] = f_val
                    map_D[y, x] = D_val
                    map_D_star[y, x] = D_star_val
                except:
                    continue

    # Başlangıç olarak f haritasını overlay göster
        self.ax.clear()
        base_img = self.image_data[int(b_fit[0])][self.current_slice_index]
        self.ax.imshow(base_img, cmap='gray')
        overlay = self.ax.imshow(map_f, cmap='jet', alpha=0.5)
        self.ax.set_title("IVIM BI Fit - f Haritası")
        self.ax.axis('off')
        self.image_canvas.draw()

    # Haritaları nesne içinde sakla
        self.map_f = map_f
        self.map_D = map_D
        self.map_D_star = map_D_star

    def display_param_map(self, param_map, title="Parametre Haritası"):
        b_vals = np.array(self.b_values)
        if self.exclude_b0_checkbox.isChecked():
            b_vals = b_vals[b_vals > 0]  # b=0 çık
        b_fit = b_vals[b_vals <= 1000]
        base_img = self.image_data[int(b_fit[0])][self.current_slice_index]

        self.ax.clear()
        self.ax.imshow(base_img, cmap='gray')
        self.ax.imshow(param_map, cmap='jet', alpha=0.5)
        self.ax.set_title(title)
        self.ax.axis('off')
        self.image_canvas.draw()

    def create_param_map_by_type(self):
        selected = self.map_type_combo.currentText()

        if selected == "ADC":
            self.create_param_map()

        elif selected in ["IVIM_f", "IVIM_D", "IVIM_D*"]:
        # IVIM haritalarını her zaman yeniden hesapla
            self.create_param_map_bi()

            if selected == "IVIM_f":
                self.display_param_map(self.map_f, "IVIM f Haritası")
            elif selected == "IVIM_D":
                self.display_param_map(self.map_D, "IVIM D Haritası")
            elif selected == "IVIM_D*":
                self.display_param_map(self.map_D_star, "IVIM D* Haritası")

        elif selected.startswith("IVIM_SEG_"):
            self.create_param_map_bis()
            if selected == "IVIM_SEG_f":
                self.display_param_map(self.map_f_seg, "Segmentli f Haritası")
            elif selected == "IVIM_SEG_D":
                 self.display_param_map(self.map_D_seg, "Segmentli D Haritası")
            elif selected == "IVIM_SEG_D*":
                 self.display_param_map(self.map_D_star_seg, "Segmentli D* Haritası")


    def apply_colormap_scale(self):
        selected = self.map_type_combo.currentText()

    # Harita seç
        if selected == "ADC":
            param_map = getattr(self, "map_adc", None)
            title = "ADC Haritası"
        elif selected == "IVIM_f":
            param_map = getattr(self, "map_f", None)
            title = "IVIM f Haritası"
        elif selected == "IVIM_D":
            param_map = getattr(self, "map_D", None)
            title = "IVIM D Haritası"
        elif selected == "IVIM_D*":
            param_map = getattr(self, "map_D_star", None)
            title = "IVIM D* Haritası"
        else:
            return

        if param_map is None:
            print("Harita yok.")
            return

        try:
            vmin = float(self.vmin_input.text())
            vmax = float(self.vmax_input.text())
        except ValueError:
            print("Geçersiz min/max değeri.")
            return

    # Görselleştir
        b_vals = np.array(self.b_values)
        if self.exclude_b0_checkbox.isChecked():
            b_vals = b_vals[b_vals > 0]  # b=0 çık
        b_fit = b_vals[b_vals <= 1000]
        base_img = self.image_data[int(b_fit[0])][self.current_slice_index]

        self.ax.clear()
        self.ax.imshow(base_img, cmap='gray')
        self.ax.imshow(param_map, cmap='jet', alpha=0.5, vmin=vmin, vmax=vmax)
        self.ax.set_title(f"{title} (vmin={vmin}, vmax={vmax})")
        self.ax.axis('off')
        self.image_canvas.draw()
    def create_param_map_bis(self):
   

        if self.roi_mask is None:
            print("ROI çizilmemiş.")
            return

        

        b_vals = np.array(self.b_values)
        if self.exclude_b0_checkbox.isChecked():
            b_vals = b_vals[b_vals > 0]  # b=0 çık
        mask = b_vals <= 1000
        b_fit = b_vals[mask]

        if len(b_fit) < 4:
            print("Yeterli b-değeri yok.")
            return

        img_shape = self.image_data[int(b_fit[0])][self.current_slice_index].shape
        map_f = np.zeros(img_shape, dtype=np.float32)
        map_D = np.zeros(img_shape, dtype=np.float32)
        map_D_star = np.zeros(img_shape, dtype=np.float32)

        for y in range(img_shape[0]):
            for x in range(img_shape[1]):
                if not self.roi_mask[y, x]:
                    continue

                signal_pixel = []
                for b in b_fit:
                    img = self.image_data[int(b)][self.current_slice_index]
                    signal_pixel.append(img[y, x])
                signal_pixel = np.array(signal_pixel)

                if signal_pixel[0] <= 0 or np.any(signal_pixel <= 0):
                    continue

                signal_norm = signal_pixel / signal_pixel[0]

                # AŞAMA 1: D tahmini (b > 200)
                b_high_mask = b_fit > 200
                if np.sum(b_high_mask) < 2:
                    continue

                X_high = -b_fit[b_high_mask].reshape(-1, 1)
                y_high = np.log(signal_norm[b_high_mask] + 1e-8)
                try:
                    D_est = np.linalg.lstsq(X_high, y_high, rcond=None)[0][0]
                except:
                    continue

                # AŞAMA 2: f ve D* tahmini (D sabit)
                def fit_func(b, f, D_star):
                    return ivim_bis_model(b, f, D_star, D_est)

                try:
                    p0 = [0.1, 0.01]
                    bounds = ([0.01, 1e-3], [0.5, 0.1])
                    popt, _ = curve_fit(fit_func, b_fit, signal_norm, p0=p0, bounds=bounds)
                    f_val, D_star_val = popt
                    map_f[y, x] = f_val
                    map_D[y, x] = D_est
                    map_D_star[y, x] = D_star_val
                except:
                    continue

        # Görsel olarak f haritasını göster
        self.ax.clear()
        base_img = self.image_data[int(b_fit[0])][self.current_slice_index]
        self.ax.imshow(base_img, cmap='gray')
        overlay = self.ax.imshow(map_f, cmap='jet', alpha=0.5)
        self.ax.set_title("IVIM Segmentli - f Haritası")
        self.ax.axis('off')
        self.image_canvas.draw()

        # Segmentli haritaları sakla
        self.map_f_seg = map_f
        self.map_D_seg = map_D
        self.map_D_star_seg = map_D_star

    def clear_roi(self):
        self.roi_mask = None

        # Görsel alanı temizle ve baz görüntüyü yeniden göster
        b_vals = np.array(self.b_values)
        b_fit = b_vals[b_vals <= 1000]
        base_img = self.image_data[int(b_fit[0])][self.current_slice_index]

        self.ax.clear()
        self.ax.imshow(base_img, cmap='gray')
        self.ax.set_title("ROI Temizlendi")
        self.ax.axis('off')
        self.image_canvas.draw()

        # (Opsiyonel) sinyal grafiğini temizle
        self.graph_ax.clear()
        self.graph_ax.set_title("Sinyal Grafiği")
        self.graph_canvas.draw()


    def export_results_to_excel(self):
        dialog = PatientInfoDialog()
        if dialog.exec_() == QDialog.Accepted:
            patient_first_name, patient_last_name = dialog.get_patient_info()

            data = {
                "Hasta Adı": [patient_first_name],
                "Soyadı": [patient_last_name],
                "ADC": [getattr(self, "adc", 0.0)],
                "Free f": [getattr(self, "free_f", 0.0)],
                "Free D": [getattr(self, "free_D", 0.0)],
                "Free D*": [getattr(self, "free_D_star", 0.0)],
                "Segment f": [getattr(self, "seg_f", 0.0)],
                "Segment D": [getattr(self, "seg_D", 0.0)],
                "Segment D*": [getattr(self, "seg_D_star", 0.0)],
                "Bayes f": [getattr(self, "bayes_f", 0.0)],
                "Bayes D": [getattr(self, "bayes_D", 0.0)],
                "Bayes D*": [getattr(self, "bayes_D_star", 0.0)],
            }

            df = pd.DataFrame(data)

           
            today_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
            filename = f"{patient_first_name}_{patient_last_name}_{today_str}.xlsx"
            output_path = os.path.join("outputs", filename)
            os.makedirs("outputs", exist_ok=True)

            df.to_excel(output_path, index=False)
            print(f"Excel dosyası oluşturuldu: {output_path}")

   
















