from PyQt5 import QtCore, QtGui, QtWidgets
import os, cv2, psutil
import numpy as np
import utils

class PhotoViewer(QtWidgets.QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, MainWindow):
        super(PhotoViewer, self).__init__(MainWindow)
        self._zoom = 0
        self._empty = True
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())
        self.fitInView()

    def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 1000)
        self.preproc_viewer = PhotoViewer(MainWindow)
        self.detect_viewer = PhotoViewer(MainWindow)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.formLayout = QtWidgets.QFormLayout(self.centralwidget)
        self.formLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setObjectName("formLayout")
        self.infos = QtWidgets.QHBoxLayout()
        self.infos.setSpacing(20)
        self.infos.setObjectName("infos")
        self.logo = QtWidgets.QLabel(self.centralwidget)
        self.logo.setObjectName('Logo')
        self.logo.setPixmap(QtGui.QPixmap(os.path.dirname(os.path.abspath(__file__)) + "\Logo.png"))
        self.infos.addWidget(self.logo)
        self.cpu = QtWidgets.QHBoxLayout()
        self.cpu.setObjectName("cpu")
        self.cpu_label = QtWidgets.QLabel(self.centralwidget)
        self.cpu_label.setObjectName("cpu_label")
        self.cpu.addWidget(self.cpu_label)
        self.cpu_bar = QtWidgets.QProgressBar(self.centralwidget)
        self.cpu_bar.setProperty("value", 1)
        self.cpu_bar.setObjectName("cpu_bar")
        self.cpu.addWidget(self.cpu_bar)
        self.infos.addLayout(self.cpu)
        self.ram = QtWidgets.QHBoxLayout()
        self.ram.setObjectName("ram")
        self.ram_label = QtWidgets.QLabel(self.centralwidget)
        self.ram_label.setObjectName("ram_label")
        self.ram.addWidget(self.ram_label)
        self.ram_bar = QtWidgets.QProgressBar(self.centralwidget)
        self.ram_bar.setProperty("value", 1)
        self.ram_bar.setObjectName("ram_bar")
        self.ram.addWidget(self.ram_bar)
        self.infos.addLayout(self.ram)
        self.formLayout.setLayout(0, QtWidgets.QFormLayout.SpanningRole, self.infos)
        self.menu = QtWidgets.QVBoxLayout()
        self.menu.setContentsMargins(15, -1, 15, -1)
        self.menu.setObjectName("menu")
        self.open = QtWidgets.QPushButton(self.centralwidget)
        self.open.setObjectName("open")
        self.open.clicked.connect(self.openfile)
        self.menu.addWidget(self.open)
        self.prepoc_button = QtWidgets.QPushButton(self.centralwidget)
        self.prepoc_button.setObjectName("prepoc_button")
        self.prepoc_button.clicked.connect(self.change_page_1)
        self.menu.addWidget(self.prepoc_button)
        self.detection_button = QtWidgets.QPushButton(self.centralwidget)
        self.detection_button.setObjectName("detection_button")
        self.detection_button.clicked.connect(self.change_page_2)
        self.menu.addWidget(self.detection_button)
        self.calc_button = QtWidgets.QPushButton(self.centralwidget)
        self.calc_button.setObjectName("calc_button")
        self.calc_button.clicked.connect(self.change_page_3)
        self.menu.addWidget(self.calc_button)
        self.reset = QtWidgets.QPushButton(self.centralwidget)
        self.reset.setObjectName("reset")
        self.menu.addWidget(self.reset)
        self.files = QtWidgets.QListWidget(self.centralwidget)
        self.files.setObjectName("files")
        self.files.currentRowChanged.connect(self.loadimage)
        self.menu.addWidget(self.files)
        self.formLayout.setLayout(1, QtWidgets.QFormLayout.LabelRole, self.menu)
        self.browser = QtWidgets.QStackedWidget(self.centralwidget)
        self.browser.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.browser.setFrameShadow(QtWidgets.QFrame.Plain)
        self.browser.setObjectName("browser")
        self.preproc = QtWidgets.QWidget()
        self.preproc.setObjectName("preproc")
        self.formLayout_2 = QtWidgets.QFormLayout(self.preproc)
        self.formLayout_2.setObjectName("formLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.auto_label = QtWidgets.QLabel(self.preproc)
        self.auto_label.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.auto_label.setAlignment(QtCore.Qt.AlignCenter)
        self.auto_label.setObjectName("auto_label")
        self.horizontalLayout.addWidget(self.auto_label)
        self.manual_label = QtWidgets.QLabel(self.preproc)
        self.manual_label.setAlignment(QtCore.Qt.AlignCenter)
        self.manual_label.setObjectName("manual_label")
        self.horizontalLayout.addWidget(self.manual_label)
        self.formLayout_2.setLayout(0, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout)
        self.preproc_tool = QtWidgets.QHBoxLayout()
        self.preproc_tool.setObjectName("preproc_tool")
        self.automatic = QtWidgets.QVBoxLayout()
        self.automatic.setObjectName("automatic")
        self.threshold_label = QtWidgets.QLabel(self.preproc)
        self.threshold_label.setObjectName("threshold_label")
        self.automatic.addWidget(self.threshold_label)
        self.threshold_box = QtWidgets.QSpinBox(self.preproc)
        self.threshold_box.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.threshold_box.setAccelerated(True)
        self.threshold_box.setMinimum(1)
        self.threshold_box.setMaximum(255)
        self.threshold_box.setStepType(QtWidgets.QAbstractSpinBox.DefaultStepType)
        self.threshold_box.setProperty("value", 60)
        self.threshold_box.setObjectName("threshold_box")
        self.automatic.addWidget(self.threshold_box)
        self.auto_apply = QtWidgets.QPushButton(self.preproc)
        self.auto_apply.setObjectName("auto_apply")
        self.auto_apply.clicked.connect(self.local_cropAuto)
        self.automatic.addWidget(self.auto_apply)
        self.preproc_tool.addLayout(self.automatic)
        self.manual = QtWidgets.QVBoxLayout()
        self.manual.setObjectName("manual")
        self.manual_settings = QtWidgets.QGridLayout()
        self.manual_settings.setContentsMargins(10, 10, 10, 10)
        self.manual_settings.setSpacing(10)
        self.manual_settings.setObjectName("manual_settings")
        self.x_max_box = QtWidgets.QSpinBox(self.preproc)
        self.x_max_box.setMaximum(10000)
        self.x_max_box.setObjectName("x_max_box")
        self.x_max_box.valueChanged.connect(self.add_xmax)
        self.manual_settings.addWidget(self.x_max_box, 1, 1, 1, 1)
        self.y_max_box = QtWidgets.QSpinBox(self.preproc)
        self.y_max_box.setObjectName("y_max_box")
        self.y_max_box.valueChanged.connect(self.add_ymax)
        self.manual_settings.addWidget(self.y_max_box, 1, 3, 1, 1)
        self.x_min_box = QtWidgets.QSpinBox(self.preproc)
        self.x_min_box.setMaximum(10000)
        self.x_min_box.setObjectName("x_min_box")
        self.x_min_box.valueChanged.connect(self.add_xmin)
        self.manual_settings.addWidget(self.x_min_box, 0, 1, 1, 1)
        self.y_min = QtWidgets.QLabel(self.preproc)
        self.y_min.setAlignment(QtCore.Qt.AlignCenter)
        self.y_min.setObjectName("y_min")
        self.manual_settings.addWidget(self.y_min, 0, 2, 1, 1)
        self.x_min = QtWidgets.QLabel(self.preproc)
        self.x_min.setAlignment(QtCore.Qt.AlignCenter)
        self.x_min.setObjectName("x_min")
        self.manual_settings.addWidget(self.x_min, 0, 0, 1, 1)
        self.x_max = QtWidgets.QLabel(self.preproc)
        self.x_max.setAlignment(QtCore.Qt.AlignCenter)
        self.x_max.setObjectName("x_max")
        self.manual_settings.addWidget(self.x_max, 1, 0, 1, 1)
        self.y_min_box = QtWidgets.QSpinBox(self.preproc)
        self.y_min_box.setObjectName("y_min_box")
        self.y_min_box.valueChanged.connect(self.add_ymin)
        self.manual_settings.addWidget(self.y_min_box, 0, 3, 1, 1)
        self.y_max = QtWidgets.QLabel(self.preproc)
        self.y_max.setAlignment(QtCore.Qt.AlignCenter)
        self.y_max.setObjectName("y_max")
        self.manual_settings.addWidget(self.y_max, 1, 2, 1, 1)
        self.manual.addLayout(self.manual_settings)
        self.manual_apply = QtWidgets.QPushButton(self.preproc)
        self.manual_apply.setObjectName("manual_apply")
        self.manual.addWidget(self.manual_apply)
        self.preproc_tool.addLayout(self.manual)
        self.formLayout_2.setLayout(1, QtWidgets.QFormLayout.FieldRole, self.preproc_tool)
        self.preproc_confirm = QtWidgets.QHBoxLayout()
        self.preproc_confirm.setObjectName("preproc_confirm")
        self.preproc_accept = QtWidgets.QPushButton(self.preproc)
        self.preproc_accept.setObjectName("preproc_accept")
        self.preproc_confirm.addWidget(self.preproc_accept)
        self.preproc_reset = QtWidgets.QPushButton(self.preproc)
        self.preproc_reset.setObjectName("preproc_reset")
        self.preproc_confirm.addWidget(self.preproc_reset)
        self.formLayout_2.setLayout(3, QtWidgets.QFormLayout.FieldRole, self.preproc_confirm)
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.preproc_viewer)
        self.browser.addWidget(self.preproc)
        self.detect = QtWidgets.QWidget()
        self.detect.setObjectName("detect_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.detect)
        self.verticalLayout.setObjectName("verticalLayout")
        self.detect_tool = QtWidgets.QHBoxLayout()
        self.detect_tool.setObjectName("detect_tool")
        self.detect_model = QtWidgets.QHBoxLayout()
        self.detect_model.setObjectName("detect_model")
        self.choose_model_label = QtWidgets.QLabel(self.detect)
        self.choose_model_label.setObjectName("choose_model_label")
        self.detect_model.addWidget(self.choose_model_label)
        self.choose_model = QtWidgets.QComboBox(self.detect)
        self.choose_model.setEditable(False)
        self.choose_model.setCurrentText("1.0")
        self.choose_model.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContentsOnFirstShow)
        self.choose_model.setPlaceholderText("")
        self.choose_model.setFrame(True)
        self.choose_model.setObjectName("choose_model")
        self.choose_model.addItem("")
        self.detect_model.addWidget(self.choose_model)
        self.detect_tool.addLayout(self.detect_model)
        self.detect_conf = QtWidgets.QGridLayout()
        self.detect_conf.setObjectName("detect_conf")
        self.show_conf = QtWidgets.QRadioButton(self.detect)
        self.show_conf.setText("")
        self.show_conf.setChecked(True)
        self.show_conf.setObjectName("show_conf")
        self.detect_conf.addWidget(self.show_conf, 1, 1, 1, 1)
        self.show_conf_label = QtWidgets.QLabel(self.detect)
        self.show_conf_label.setObjectName("show_conf_label")
        self.detect_conf.addWidget(self.show_conf_label, 1, 0, 1, 1)
        self.taux_conf_label = QtWidgets.QLabel(self.detect)
        self.taux_conf_label.setObjectName("taux_conf_label")
        self.detect_conf.addWidget(self.taux_conf_label, 0, 0, 1, 1)
        self.taux_conf = QtWidgets.QDoubleSpinBox(self.detect)
        self.taux_conf.setDecimals(1)
        self.taux_conf.setMaximum(100.0)
        self.taux_conf.setStepType(QtWidgets.QAbstractSpinBox.DefaultStepType)
        self.taux_conf.setProperty("value", 95.0)
        self.taux_conf.setObjectName("taux_conf")
        self.detect_conf.addWidget(self.taux_conf, 0, 1, 1, 1)
        self.detect_tool.addLayout(self.detect_conf)
        self.detect_box = QtWidgets.QHBoxLayout()
        self.detect_box.setObjectName("detect_box")
        self.box_width_label = QtWidgets.QLabel(self.detect)
        self.box_width_label.setObjectName("box_width_label")
        self.detect_box.addWidget(self.box_width_label)
        self.box_width = QtWidgets.QDoubleSpinBox(self.detect)
        self.box_width.setDecimals(1)
        self.box_width.setProperty("value", 1.0)
        self.box_width.setObjectName("box_width")
        self.detect_box.addWidget(self.box_width)
        self.detect_tool.addLayout(self.detect_box)
        self.start_detect = QtWidgets.QPushButton(self.detect)
        self.start_detect.setObjectName("start_detect")
        self.detect_tool.addWidget(self.start_detect)
        self.verticalLayout.addLayout(self.detect_tool)
        self.verticalLayout.addWidget(self.detect_viewer)
        self.browser.addWidget(self.detect)
        self.calculs = QtWidgets.QWidget()
        self.calculs.setObjectName("calculs")
        self.browser.addWidget(self.calculs)
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.browser)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1092, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.browser.setCurrentIndex(0)
        self.choose_model.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.timer = QtCore.QTimer(interval=1000, timeout=self.update_usages)
        self.timer.start()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.cpu_label.setText(_translate("MainWindow", "CPU"))
        self.ram_label.setText(_translate("MainWindow", "RAM"))
        self.open.setWhatsThis(_translate("MainWindow", "<html><head/><body><p>Ouvrir un image ou un dossier</p></body></html>"))
        self.open.setText(_translate("MainWindow", "Ouvrir"))
        self.prepoc_button.setText(_translate("MainWindow", "Prétraitement"))
        self.detection_button.setText(_translate("MainWindow", "Détéction"))
        self.calc_button.setText(_translate("MainWindow", "Calculs"))
        self.reset.setText(_translate("MainWindow", "Reset"))
        self.auto_label.setText(_translate("MainWindow", "Automatic"))
        self.manual_label.setText(_translate("MainWindow", "Manuel"))
        self.threshold_label.setText(_translate("MainWindow", "Valeur threshold (0 - 255)"))
        self.auto_apply.setText(_translate("MainWindow", "Appliquer"))
        self.y_min.setText(_translate("MainWindow", "Y Min"))
        self.x_min.setText(_translate("MainWindow", "X Min"))
        self.x_max.setText(_translate("MainWindow", "X Max"))
        self.y_max.setText(_translate("MainWindow", "Y Max"))
        self.manual_apply.setText(_translate("MainWindow", "Appliquer"))
        self.preproc_accept.setText(_translate("MainWindow", "Confirmer"))
        self.preproc_reset.setText(_translate("MainWindow", "Annuler"))
        self.choose_model_label.setText(_translate("MainWindow", "Choix du modèle"))
        self.choose_model.setItemText(0, _translate("MainWindow", "1.0"))
        self.show_conf_label.setText(_translate("MainWindow", "Montrer la confiance"))
        self.taux_conf_label.setText(_translate("MainWindow", "Intervalle de confiance"))
        self.taux_conf.setSuffix(_translate("MainWindow", "%"))
        self.box_width_label.setText(_translate("MainWindow", "Épaisseur des boites"))
        self.start_detect.setText(_translate("MainWindow", "Lancer la détéction"))
    
    def change_page_1(self):
        self.browser.setCurrentIndex(0)

    def change_page_2(self):
        self.browser.setCurrentIndex(1)

    def change_page_3(self):
        self.browser.setCurrentIndex(2)

    def openfile(self):
        self.folder = str(QtWidgets.QFileDialog.getExistingDirectory(MainWindow, "Sélectionner un dossier"))
        if self.folder:
            self.files.clear()
            for file in os.listdir(self.folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.tif')):
                    self.files.addItem(file)
    
    def loadimage(self):
        self.image = cv2.imdecode(np.fromfile(self.folder + '/' + self.files.currentItem().text(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.set_image_from_cv()

    def set_image_from_cv (self):
        height, width, channel = self.image.shape
        bytesPerLine = channel * width
        qImg = QtGui.QImage(self.image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        self.detect_viewer.setPhoto(QtGui.QPixmap(qImg))
        self.preproc_viewer.setPhoto(QtGui.QPixmap(qImg))

    def update_usages(self):
        self.cpu_bar.setValue(psutil.cpu_percent())
        self.ram_bar.setValue(psutil.virtual_memory().percent)
    
    def local_cropAuto(self):
        self.loadimage()
        self.image = utils.cropAuto(self.image, self.threshold_box.value())
        self.set_image_from_cv()

    def add_xmax(self):
        self.image = cv2.line(self.image, (self.x_max_box.value(), 0), (self.x_max_box.value(), self.image.shape[0]), (255, 0, 0), 2) 
        self.set_image_from_cv()

    def add_xmin(self):
        print('xmin')

    def add_ymax(self):
        print('')

    def add_ymin(self):
        print('xmin')

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)

    app.setStyle('Fusion')

    app.setWindowIcon(QtGui.QIcon(os.path.dirname(os.path.abspath(__file__)) + "\Image.png"))

    # app.setStyleSheet("""
    #                     QMainWindow {
    #                         background-color: #0e2f44;
    #                     }
    #                     QDialog {
    #                         background-color: #0e2f44;
    #                     }
    #                     QLabel {
    #                         color: #afeeee;
    #                         font-family: "Bahnschrift";
    #                     }
    #                     QPushButton {
    #                         color : #afeeee;
    #                         background-color: #8b0000;
    #                         border-style: outset;
    #                         border-width: 2px;
    #                         border-radius: 10px;
    #                         border-color: #6897bb;
    #                         font: bold 16px;
    #                         min-width: 5em;
    #                         padding: 6px;
    #                     }
    #                     QPushButton:disabled {
    #                         color : #101010;
    #                         background-color : #101010;
    #                     }
    #                     QPushButton:pressed {
    #                         background-color: #800000;
    #                         border-style: inset;
    #                     }
    #                     QSpinBox {
    #                         background-color: #6897bb;
    #                     }
    #                     QComboBox {
    #                         background-color: #6897bb;
    #                     }
    #                     QTextBrowser{
    #                         background-color: #6897bb;
    #                     }
    #                     QTextEdit {
    #                         background-color: #6897bb;
    #                     }
    #                     QLineEdit {
    #                         background-color: #6897bb;
    #                     }

    #                 """)

    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())