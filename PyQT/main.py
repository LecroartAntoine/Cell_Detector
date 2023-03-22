from PyQt5 import QtCore, QtGui, QtWidgets
import os, cv2, psutil, ImageViewer, sys
import numpy as np
import utils

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 800)

        self.image = np.zeros((0,0,0), np.uint8)

        # Image Viewer
        self.preproc_viewer = ImageViewer.PhotoViewer(MainWindow)
        self.detect_viewer = ImageViewer.PhotoViewer(MainWindow)

        # Main window
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.formLayout = QtWidgets.QFormLayout(self.centralwidget)
        self.formLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout.setObjectName("formLayout")

#####################################  Infos  ########################################################
        self.infos = QtWidgets.QHBoxLayout()
        self.infos.setSpacing(20)
        self.infos.setObjectName("infos")
        self.logo = QtWidgets.QLabel(self.centralwidget)
        self.logo.setObjectName('Logo')
        self.logo.setPixmap(QtGui.QPixmap(os.path.dirname(os.path.abspath(__file__)) + "\Assets\Logo.png"))
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

#################################  MENU  ##################################################
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
        self.calc_button.setEnabled(False)
        self.menu.addWidget(self.calc_button)
        self.reset = QtWidgets.QPushButton(self.centralwidget)
        self.reset.setObjectName("reset")
        self.menu.addWidget(self.reset)
        self.files = QtWidgets.QListWidget(self.centralwidget)
        self.files.setObjectName("files")
        self.files.currentRowChanged.connect(self.loadimage)
        self.menu.addWidget(self.files)
        self.formLayout.setLayout(1, QtWidgets.QFormLayout.LabelRole, self.menu)


#####################################  Browser  ########################################################
        self.browser = QtWidgets.QStackedWidget(self.centralwidget)
        self.browser.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.browser.setFrameShadow(QtWidgets.QFrame.Plain)
        self.browser.setObjectName("browser")
        self.browser.setCurrentIndex(0)

#####################################  Preprocessing  ########################################################
        self.preproc = QtWidgets.QWidget()
        self.preproc.setObjectName("preproc")
        self.formLayout_2 = QtWidgets.QFormLayout(self.preproc)
        self.formLayout_2.setObjectName("formLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")

#####################################  Auto crop  ########################################################
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

######################## Manual crop #####################################
        self.manual = QtWidgets.QVBoxLayout()
        self.manual.setObjectName("manual")
        self.manual_settings = QtWidgets.QGridLayout()
        self.manual_settings.setContentsMargins(10, 10, 10, 10)
        self.manual_settings.setSpacing(10)
        self.manual_settings.setObjectName("manual_settings")

        # x min
        self.x_min = QtWidgets.QLabel(self.preproc)
        self.x_min.setAlignment(QtCore.Qt.AlignCenter)
        self.x_min.setObjectName("x_min")
        self.manual_settings.addWidget(self.x_min, 0, 0, 1, 1)
        self.x_min_box = QtWidgets.QSpinBox(self.preproc)
        self.x_min_box.setSingleStep(10)
        self.x_min_box.setMaximum(0)
        self.x_min_box.setObjectName("x_min_box")
        self.x_min_box.valueChanged.connect(self.add_x_min_line)
        self.x_min_box.setAccelerated(True)
        self.manual_settings.addWidget(self.x_min_box, 0, 1, 1, 1)

        # x max
        self.x_max = QtWidgets.QLabel(self.preproc)
        self.x_max.setAlignment(QtCore.Qt.AlignCenter)
        self.x_max.setObjectName("x_max")
        self.manual_settings.addWidget(self.x_max, 1, 0, 1, 1)
        self.x_max_box = QtWidgets.QSpinBox(self.preproc)
        self.x_max_box.setSingleStep(10)
        self.x_max_box.setMaximum(0)
        self.x_max_box.setObjectName("x_max_box")
        self.x_max_box.valueChanged.connect(self.add_x_max_line)
        self.x_max_box.setAccelerated(True)
        self.manual_settings.addWidget(self.x_max_box, 1, 1, 1, 1)

        # y min
        self.y_min = QtWidgets.QLabel(self.preproc)
        self.y_min.setAlignment(QtCore.Qt.AlignCenter)
        self.y_min.setObjectName("y_min")
        self.manual_settings.addWidget(self.y_min, 0, 2, 1, 1)
        self.y_min_box = QtWidgets.QSpinBox(self.preproc)
        self.y_min_box.setSingleStep(10)
        self.y_min_box.setMaximum(0)
        self.y_min_box.setObjectName("y_min_box")
        self.y_min_box.valueChanged.connect(self.add_y_min_line)
        self.y_min_box.setAccelerated(True)
        self.manual_settings.addWidget(self.y_min_box, 0, 3, 1, 1)

        # y max
        self.y_max = QtWidgets.QLabel(self.preproc)
        self.y_max.setAlignment(QtCore.Qt.AlignCenter)
        self.y_max.setObjectName("y_max")
        self.manual_settings.addWidget(self.y_max, 1, 2, 1, 1)
        self.y_max_box = QtWidgets.QSpinBox(self.preproc)
        self.y_max_box.setSingleStep(10)
        self.y_max_box.setMaximum(0)
        self.y_max_box.setObjectName("y_max_box")
        self.y_max_box.valueChanged.connect(self.add_y_max_line)
        self.y_max_box.setAccelerated(True)
        self.manual_settings.addWidget(self.y_max_box, 1, 3, 1, 1)


        self.manual.addLayout(self.manual_settings)
        self.manual_apply = QtWidgets.QPushButton(self.preproc)
        self.manual_apply.setObjectName("manual_apply")
        self.manual_apply.clicked.connect(self.manual_crop)
        self.manual.addWidget(self.manual_apply)
        self.preproc_tool.addLayout(self.manual)

        self.formLayout_2.setLayout(1, QtWidgets.QFormLayout.FieldRole, self.preproc_tool)
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.preproc_viewer)
        self.browser.addWidget(self.preproc)

#####################################  Yolo detect  ########################################################
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
        self.choose_model.setCurrentIndex(0)
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
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.detect_tool.addItem(spacerItem)
        self.verticalLayout.addLayout(self.detect_tool)
        self.verticalLayout.addWidget(self.detect_viewer)


        self.browser.addWidget(self.detect)
        self.calculs = QtWidgets.QWidget()
        self.calculs.setObjectName("calculs")
        self.browser.addWidget(self.calculs)
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.browser)

#####################################  div  ########################################################
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1092, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.timer = QtCore.QTimer(interval=1000, timeout=self.update_usages)
        self.timer.start()

        MainWindow.showMaximized()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        # MainWindow.setWindowTitle(_translate("MainWindow", "a batiser"))
        self.cpu_label.setText(_translate("MainWindow", "CPU"))
        self.ram_label.setText(_translate("MainWindow", "RAM"))
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
        self.choose_model_label.setText(_translate("MainWindow", "Choix du modèle"))
        self.choose_model.setItemText(0, _translate("MainWindow", "1.0"))
        self.show_conf_label.setText(_translate("MainWindow", "Montrer la confiance"))
        self.taux_conf_label.setText(_translate("MainWindow", "Intervalle de confiance"))
        self.taux_conf.setSuffix(_translate("MainWindow", "%"))
        self.box_width_label.setText(_translate("MainWindow", "Épaisseur des boites"))
        self.start_detect.setText(_translate("MainWindow", "Lancer la détéction"))
    
    def change_page_1(self):
        self.browser.setCurrentIndex(0)
        self.set_image_from_cv()

    def change_page_2(self):
        self.browser.setCurrentIndex(1)
        self.set_image_from_cv()

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
        self.x_min_box.setMaximum(self.image.shape[1])
        self.x_max_box.setMaximum(self.image.shape[1])
        self.y_min_box.setMaximum(self.image.shape[0])
        self.y_max_box.setMaximum(self.image.shape[0])
        self.set_image_from_cv()

    def set_image_from_cv (self):
        height, width, channel = self.image.shape
        bytesPerLine = channel * width
        qImg = QtGui.QImage(self.image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        self.detect_viewer.setPhoto(QtGui.QPixmap(qImg))
        self.preproc_viewer.setPhoto(QtGui.QPixmap(qImg))

    def update_usages(self):
        self.cpu_bar.setValue(int(psutil.cpu_percent()))
        self.ram_bar.setValue(int(psutil.virtual_memory().percent))
    
    def local_cropAuto(self):
        try:
            self.loadimage()
            self.image = utils.cropAuto(self.image, self.threshold_box.value())
            self.set_image_from_cv()
        except:
            pass

    def add_x_max_line(self):

        try:
            if len(self.preproc_viewer.scene().items()) > 1:
                self.preproc_viewer.scene().removeItem(self.preproc_viewer.scene().items()[0])
            self.preproc_viewer.scene().addLine(self.x_max_box.value(), 0, self.x_max_box.value(), self.image.shape[0], pen = QtGui.QPen(QtGui.QBrush(QtGui.QColor(255, 0, 0) ), 5))
        except:
            pass

    def add_x_min_line(self):
        try:
            if len(self.preproc_viewer.scene().items()) > 1:
                self.preproc_viewer.scene().removeItem(self.preproc_viewer.scene().items()[0])
            self.preproc_viewer.scene().addLine(self.x_min_box.value(), 0, self.x_min_box.value(), self.image.shape[0], pen = QtGui.QPen(QtGui.QBrush(QtGui.QColor(255, 0, 0) ), 5))
        except:
            pass

    def add_y_max_line(self):
        try:
            if len(self.preproc_viewer.scene().items()) > 1:
                self.preproc_viewer.scene().removeItem(self.preproc_viewer.scene().items()[0])
            self.preproc_viewer.scene().addLine(0, self.y_max_box.value(), self.image.shape[1], self.y_max_box.value(), pen = QtGui.QPen(QtGui.QBrush(QtGui.QColor(255, 0, 0) ), 5))
        except:
            pass

    def add_y_min_line(self):
        try:
            if len(self.preproc_viewer.scene().items()) > 1:
                self.preproc_viewer.scene().removeItem(self.preproc_viewer.scene().items()[0])
            self.preproc_viewer.scene().addLine(0, self.y_min_box.value(), self.image.shape[1], self.y_min_box.value(), pen = QtGui.QPen(QtGui.QBrush(QtGui.QColor(255, 0, 0) ), 5))
        except:
            pass
    
    def manual_crop(self):
        try:
            if len(self.preproc_viewer.scene().items()) > 1:
                self.preproc_viewer.scene().removeItem(self.preproc_viewer.scene().items()[0])
            self.loadimage()
            self.image = self.image[self.y_min_box.value() : self.y_max_box.value(), self.x_min_box.value() : self.x_max_box.value()].copy()
            x = self.image.shape[0] if self.image.shape[0] > self.image.shape[1] else self.image.shape[1]
            while x % 32 != 0:
                x -= 1
            self.image = cv2.resize(self.image, (x, x), interpolation = cv2.INTER_AREA)
            self.set_image_from_cv()
        except:
            pass


if __name__ == "__main__":

    path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

    app = QtWidgets.QApplication(sys.argv)

    app.setStyle('Fusion')

    app.setWindowIcon(QtGui.QIcon(path + "/Assets/Logo_small.png"))

    app.setStyleSheet(f"""
                        QMainWindow {{
                            background-image: url({path}/Assets/Fond.png);
                        }}
                        QLabel {{
                        
                        }}
                        QPushButton {{
                        }}
                        QPushButton:disabled {{
                        }}
                        QPushButton:pressed {{
                        }}
                        QSpinBox {{
                        }}
                        QComboBox {{
                        }}
                    """)

    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
