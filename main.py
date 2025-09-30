import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from gui.tab_dicom import DICOMLoaderTab
from gui.tab_diffusion import DiffusionViewerTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("DICOM & Difüzyon Arayüzü")
        self.setGeometry(100, 100, 1600, 1000)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Sekmeleri oluştur
        self.tab2 = DiffusionViewerTab()
        self.tab1 = DICOMLoaderTab(parent=self)  # Ana pencere referansı veriyoruz

        self.tabs.addTab(self.tab1, "DICOM Yükle")
        self.tabs.addTab(self.tab2, "Difüzyon İncele")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
