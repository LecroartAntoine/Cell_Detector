from PyQt5 import QtCore, QtGui, QtWidgets
import os, cv2, psutil, ImageViewer, sys
import numpy as np
import utils

class DetectThread(QtCore.QThread):
    finished = QtCore.pyqtSignal(object)

    def __init__(self, image, taux_conf, box_width, show_conf, show_name, parent=None):
        super().__init__(parent)
        self.image = image
        self.taux_conf = taux_conf
        self.box_width = box_width
        self.show_conf = show_conf
        self.show_name = show_name

    def run(self):
        pred = utils.yolo_detection(self.image, self.taux_conf / 100)
        image_pred = utils.plot_bboxes(self.image, pred.boxes.boxes, self.box_width, self.show_conf, self.show_name)
        self.finished.emit((pred, image_pred))

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 800)

        self.image = np.zeros((0,0,0), np.uint8)

        # Image Viewer
        self.detect_viewer = ImageViewer.PhotoViewer(MainWindow)
        self.analyse_viewer = ImageViewer.PhotoViewer(MainWindow)

        # Main window
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.mainLayout = QtWidgets.QFormLayout(self.centralwidget)
        self.mainLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.mainLayout.setObjectName("mainLayout")

#####################################  Infos  ########################################################
        self.infos = QtWidgets.QHBoxLayout()
        self.infos.setSpacing(20)
        self.infos.setObjectName("infos")
        self.logo = QtWidgets.QLabel(self.centralwidget)
        self.logo.setObjectName('Logo')
        self.logo.setPixmap(QtGui.QPixmap("Assets:Logo.png"))
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
        self.mainLayout.setLayout(0, QtWidgets.QFormLayout.SpanningRole, self.infos)

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
        self.prepoc_button.setEnabled(False)
        self.menu.addWidget(self.prepoc_button)

        self.detection_button = QtWidgets.QPushButton(self.centralwidget)
        self.detection_button.setObjectName("detection_button")
        self.detection_button.clicked.connect(self.change_page_2)
        self.detection_button.setEnabled(False)
        self.menu.addWidget(self.detection_button)

        self.analyse_button = QtWidgets.QPushButton(self.centralwidget)
        self.analyse_button.setObjectName("analyse_button")
        self.analyse_button.clicked.connect(self.change_page_3)
        self.analyse_button.setEnabled(False)
        self.menu.addWidget(self.analyse_button)

        self.reset = QtWidgets.QPushButton(self.centralwidget)
        self.reset.setObjectName("reset")
        self.reset.setEnabled(False)
        self.menu.addWidget(self.reset)

        self.files = QtWidgets.QListWidget(self.centralwidget)
        self.files.setObjectName("files")
        self.files.currentRowChanged.connect(self.loadimage)
        self.menu.addWidget(self.files)
        self.mainLayout.setLayout(1, QtWidgets.QFormLayout.LabelRole, self.menu)


#####################################  Browser  ########################################################
        self.browser = QtWidgets.QStackedWidget(self.centralwidget)
        self.browser.setObjectName("browser")
        self.browser.setCurrentIndex(0)

##################################### Welcome page ########################################################

        self.welcome = QtWidgets.QWidget()
        self.welcome.setObjectName("welcome")
        self.welcome_layout = QtWidgets.QVBoxLayout(self.welcome)
        self.welcome_layout.setObjectName("welcome_layout")

        self.welcome_message_1 = QtWidgets.QLabel(self.welcome)
        self.welcome_message_1.setAlignment(QtCore.Qt.AlignCenter)
        self.welcome_message_1.setObjectName("welcome_message_1")

        self.welcome_message_2 = QtWidgets.QLabel(self.welcome)
        self.welcome_message_2.setAlignment(QtCore.Qt.AlignCenter)
        self.welcome_message_2.setObjectName("welcome_message_2")

        self.welcome_layout.addWidget(self.welcome_message_1)
        self.welcome_layout.addWidget(self.welcome_message_2)
        self.browser.addWidget(self.welcome)

#####################################  Preprocessing  ########################################################
        self.select = QtWidgets.QWidget()
        self.select.setObjectName("select")
        self.image_select_layout = QtWidgets.QVBoxLayout(self.select)
        self.image_select_layout.setContentsMargins(0, 0, 0, 0)
        self.image_select_layout.setObjectName("image_select_layout")
        self.image_type_layout = QtWidgets.QGridLayout()
        self.image_type_layout.setObjectName("image_type_layout")
        self.type_2s = QtWidgets.QComboBox(self.select)
        self.type_2s.setObjectName("type_2s")
        self.image_type_layout.addWidget(self.type_2s, 3, 1, 1, 1)
        self.type_1c = QtWidgets.QComboBox(self.select)
        self.type_1c.setObjectName("type_1c")
        self.image_type_layout.addWidget(self.type_1c, 6, 0, 1, 1)
        self.type2c = QtWidgets.QComboBox(self.select)
        self.type2c.setObjectName("type2c")
        self.image_type_layout.addWidget(self.type2c, 6, 1, 1, 1)
        self.type_1s = QtWidgets.QComboBox(self.select)
        self.type_1s.setObjectName("type_1s")
        self.image_type_layout.addWidget(self.type_1s, 3, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.image_type_layout.addItem(spacerItem, 4, 0, 1, 1)
        self.type_2c_label = QtWidgets.QLabel(self.select)
        self.type_2c_label.setObjectName("type_2c_label")
        self.image_type_layout.addWidget(self.type_2c_label, 5, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.image_type_layout.addItem(spacerItem1, 4, 1, 1, 1)
        self.type_2s_label = QtWidgets.QLabel(self.select)
        self.type_2s_label.setObjectName("type_2s_label")
        self.image_type_layout.addWidget(self.type_2s_label, 2, 1, 1, 1)
        self.type_3c = QtWidgets.QComboBox(self.select)
        self.type_3c.setObjectName("type_3c")
        self.image_type_layout.addWidget(self.type_3c, 6, 2, 1, 1)
        self.type_1c_label = QtWidgets.QLabel(self.select)
        self.type_1c_label.setObjectName("type_1c_label")
        self.image_type_layout.addWidget(self.type_1c_label, 5, 0, 1, 1)
        self.type_1s_label = QtWidgets.QLabel(self.select)
        self.type_1s_label.setObjectName("type_1s_label")
        self.image_type_layout.addWidget(self.type_1s_label, 2, 0, 1, 1)
        self.type_3s_label = QtWidgets.QLabel(self.select)
        self.type_3s_label.setObjectName("type_3s_label")
        self.image_type_layout.addWidget(self.type_3s_label, 2, 2, 1, 1)
        self.type_3s = QtWidgets.QComboBox(self.select)
        self.type_3s.setObjectName("type_3s")
        self.image_type_layout.addWidget(self.type_3s, 3, 2, 1, 1)
        self.type_3c_label = QtWidgets.QLabel(self.select)
        self.type_3c_label.setObjectName("type_3c_label")
        self.image_type_layout.addWidget(self.type_3c_label, 5, 2, 1, 1)
        self.image_select_label = QtWidgets.QLabel(self.select)
        self.image_select_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_select_label.setObjectName("image_select_label")
        self.image_type_layout.addWidget(self.image_select_label, 0, 1, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.image_type_layout.addItem(spacerItem2, 4, 2, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.image_type_layout.addItem(spacerItem3, 1, 1, 1, 1)
        self.image_type_layout.setRowMinimumHeight(0, 60)
        self.image_type_layout.setRowMinimumHeight(1, 60)
        self.image_type_layout.setRowMinimumHeight(2, 60)
        self.image_type_layout.setRowMinimumHeight(3, 60)
        self.image_type_layout.setRowMinimumHeight(4, 60)
        self.image_select_layout.addLayout(self.image_type_layout)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.image_select_layout.addItem(spacerItem4)
        self.image_select_confirm_layout = QtWidgets.QHBoxLayout()
        self.image_select_confirm_layout.setObjectName("image_select_confirm_layout")
        self.image_select_confirm = QtWidgets.QPushButton(self.select)
        self.image_select_confirm.setObjectName("image_select_confirm")
        self.image_select_confirm_layout.addWidget(self.image_select_confirm)
        self.image_select_deny = QtWidgets.QPushButton(self.select)
        self.image_select_deny.setObjectName("image_select_deny")
        self.image_select_confirm_layout.addWidget(self.image_select_deny)
        self.image_select_layout.addLayout(self.image_select_confirm_layout)
        self.browser.addWidget(self.select)

#####################################  Yolo detect  ########################################################
        self.detect = QtWidgets.QWidget()
        self.detect.setObjectName("detect")
        self.detect_layout = QtWidgets.QVBoxLayout(self.detect)
        self.detect_layout.setObjectName("detect_layout")
        self.detect_tool = QtWidgets.QHBoxLayout()
        self.detect_tool.setObjectName("detect_tool")
        self.detect_tool.setSpacing(20)
        self.detect_model = QtWidgets.QHBoxLayout()
        self.detect_model.setObjectName("detect_model")

        self.show_conf_layout = QtWidgets.QHBoxLayout()
        self.show_conf_layout.setObjectName("show_conf_layout")
        self.show_conf_label = QtWidgets.QLabel(self.detect)
        self.show_conf_label.setObjectName("show_conf_label")
        self.show_conf_layout.addWidget(self.show_conf_label)
        self.show_conf = QtWidgets.QRadioButton(self.detect)
        self.show_conf.setObjectName("show_conf")
        self.show_conf_layout.addWidget(self.show_conf)
        self.detect_tool.addLayout(self.show_conf_layout)

        self.show_name_layout = QtWidgets.QHBoxLayout()
        self.show_name_layout.setObjectName("show_name_layout")
        self.show_name_label = QtWidgets.QLabel(self.detect)
        self.show_name_label.setObjectName("show_name_label")
        self.show_name_layout.addWidget(self.show_name_label)
        self.show_name = QtWidgets.QRadioButton(self.detect)
        self.show_name.setObjectName("show_name")
        self.show_name_layout.addWidget(self.show_name)
        self.detect_tool.addLayout(self.show_name_layout)

        self.group = QtWidgets.QButtonGroup()
        self.group.addButton(self.show_conf)
        self.group.addButton(self.show_name)       
        self.group.setExclusive(False)
        self.show_name.setChecked(True)
        self.show_conf.setChecked(True)

        self.taux_conf_layout  = QtWidgets.QHBoxLayout()
        self.taux_conf_layout.setObjectName("taux_conf_layout")
        self.taux_conf_label = QtWidgets.QLabel(self.detect)
        self.taux_conf_label.setObjectName("taux_conf_label")
        self.taux_conf_layout.addWidget(self.taux_conf_label)
        self.taux_conf = QtWidgets.QSpinBox(self.detect)
        self.taux_conf.setMaximum(100)
        self.taux_conf.setAccelerated(True)
        self.taux_conf.setProperty("value", 95)
        self.taux_conf.setObjectName("taux_conf")
        self.taux_conf_layout.addWidget(self.taux_conf)
        self.detect_tool.addLayout(self.taux_conf_layout)
        self.detect_box = QtWidgets.QHBoxLayout()
        self.detect_box.setObjectName("detect_box")
        self.box_width_label = QtWidgets.QLabel(self.detect)
        self.box_width_label.setObjectName("box_width_label")
        self.detect_box.addWidget(self.box_width_label)
        self.box_width = QtWidgets.QSpinBox(self.detect)
        self.box_width.setProperty("value", 1)
        self.box_width.setObjectName("box_width")

        self.detect_box.addWidget(self.box_width)
        self.detect_tool.addLayout(self.detect_box)
        self.start_detect = QtWidgets.QPushButton(self.detect)
        self.start_detect.setObjectName("start_detect")
        self.start_detect.clicked.connect(self.get_pred)
        self.detect_tool.addWidget(self.start_detect)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.detect_tool.addItem(spacerItem)
        self.detect_layout.addLayout(self.detect_tool)
        self.detect_layout.addWidget(self.detect_viewer)
        self.browser.addWidget(self.detect)

#####################################  Analyse  ########################################################
        self.analyse = QtWidgets.QWidget()
        self.analyse.setObjectName("analyse")

        self.analyse_layout = QtWidgets.QVBoxLayout(self.analyse)
        self.analyse_layout.setObjectName("analyse_layout")

        self.analyse_tool = QtWidgets.QHBoxLayout()
        self.analyse_tool.setObjectName("analyse_tool")
        self.analyse_tool.setSpacing(20)

        self.show_detection_button = QtWidgets.QPushButton(self.analyse)
        self.show_detection_button.setObjectName("show_detection_button")
        self.show_detection_button.clicked.connect(self.show_detection)
        self.analyse_tool.addWidget(self.show_detection_button)

        self.mallassez_button = QtWidgets.QPushButton(self.analyse)
        self.mallassez_button.setObjectName("mallassez_button")
        self.mallassez_button.clicked.connect(self.mallassez_calc)
        self.analyse_tool.addWidget(self.mallassez_button)

        self.mallassez_clean_result = QtWidgets.QLineEdit(self.analyse)
        self.mallassez_clean_result.setAlignment(QtCore.Qt.AlignCenter)
        self.mallassez_clean_result.setReadOnly(True)
        self.mallassez_clean_result.setObjectName("mallassez_clean_result")
        self.analyse_tool.addWidget(self.mallassez_clean_result)

        self.mallassez_dirty_result = QtWidgets.QLineEdit(self.analyse)
        self.mallassez_dirty_result.setAlignment(QtCore.Qt.AlignCenter)
        self.mallassez_dirty_result.setReadOnly(True)
        self.mallassez_dirty_result.setObjectName("mallassez_dirty_result")
        self.analyse_tool.addWidget(self.mallassez_dirty_result)

        self.analyse_layout.addLayout(self.analyse_tool)
        self.analyse_layout.addWidget(self.analyse_viewer)
        self.browser.addWidget(self.analyse)


        self.mainLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.browser)

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
        MainWindow.setWindowTitle(_translate("MainWindow", "à baptiser"))
        self.cpu_label.setText(_translate("MainWindow", "CPU"))
        self.ram_label.setText(_translate("MainWindow", "RAM"))
        self.open.setText(_translate("MainWindow", "Ouvrir"))
        self.prepoc_button.setText(_translate("MainWindow", "Prétraitement"))
        self.detection_button.setText(_translate("MainWindow", "Détection"))
        self.analyse_button.setText(_translate("MainWindow", "Analyse"))
        self.reset.setText(_translate("MainWindow", ""))
        self.welcome_message_1.setText(_translate("MainWindow", "Bienvenue"))
        self.welcome_message_2.setText(_translate("MainWindow", "Veuillez ouvrir un dossier contenant vos images"))
        self.image_select_label.setText(_translate("MainWindow", "Confirmer les images à traiter"))
        self.type_1s_label.setText(_translate("MainWindow", "Echantillon 1 : Surnageant"))
        self.type_2s_label.setText(_translate("MainWindow", "Echantillon 2 : Surnageant"))
        self.type_2c_label.setText(_translate("MainWindow", "Echantillon 2 : Culot"))
        self.type_3s_label.setText(_translate("MainWindow", "Echantillon 3 : Surnageant"))
        self.type_1c_label.setText(_translate("MainWindow", "Echantillon 1 : Culot"))
        self.type_3c_label.setText(_translate("MainWindow", "Echantillon 3 : Culot"))
        self.image_select_confirm.setText(_translate("MainWindow", "Valider"))
        self.image_select_deny.setText(_translate("MainWindow", "Annuler"))
        self.show_conf_label.setText(_translate("MainWindow", "Montrer la confiance"))
        self.show_name_label.setText(_translate("MainWindow", "Montrer le nom du label"))
        self.taux_conf_label.setText(_translate("MainWindow", "Intervalle de confiance"))
        self.taux_conf.setSuffix(_translate("MainWindow", "%"))
        self.box_width_label.setText(_translate("MainWindow", "Épaisseur des boites"))
        self.start_detect.setText(_translate("MainWindow", "Lancer la détection"))
        self.show_detection_button.setText(_translate("MainWindow", "Voir les cellules détectées"))
        self.mallassez_button.setText(_translate("MainWindow", "Calcul de Mallassez"))

    
    def change_page_1(self):
        self.browser.setCurrentIndex(1)

    def change_page_2(self):
        self.browser.setCurrentIndex(2)
        self.set_image_from_cv(self.image, 2)

    def change_page_3(self):
        self.browser.setCurrentIndex(3)
        self.set_image_from_cv(self.image_pred, 3)

    def openfile(self):
        self.files.clear()
        self.folder = str(QtWidgets.QFileDialog.getExistingDirectory(MainWindow, "Sélectionner un dossier"))
        if self.folder:
            for file in os.listdir(self.folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.tif')):
                    self.files.addItem(file)
        self.change_page_1()
    
    def loadimage(self):
        
        self.image = cv2.imdecode(np.fromfile(self.folder + '/' + self.files.currentItem().text(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.set_image_from_cv(self.image)

        self.prepoc_button.setEnabled(True)
        self.detection_button.setEnabled(True)
        self.analyse_button.setEnabled(False)

    

    def set_image_from_cv (self, img, page=0):
        height, width, channel = img.shape
        bytesPerLine = channel * width
        qImg = QtGui.QImage(img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)

        if page == 2:
            self.detect_viewer.setPhoto(QtGui.QPixmap(qImg))
        elif page == 3:
            self.analyse_viewer.setPhoto(QtGui.QPixmap(qImg))
        else:
            self.detect_viewer.setPhoto(QtGui.QPixmap(qImg))
            self.analyse_viewer.setPhoto(QtGui.QPixmap(qImg))

    def update_usages(self):
        self.cpu_bar.setValue(int(psutil.cpu_percent()))
        self.ram_bar.setValue(int(psutil.virtual_memory().percent))
    

    def get_pred(self):
        
        progress_dialog = QtWidgets.QProgressDialog('Détection en cours, patientez ...', None, 0, 0, MainWindow)
        progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        progress_dialog.setWindowTitle('Détection')
        progress_dialog.show()

        self.worker = DetectThread(self.image, self.taux_conf.value(), self.box_width.value(), self.show_conf.isChecked(), self.show_name.isChecked())
        self.worker.finished.connect(self.show_pred)
        self.worker.finished.connect(progress_dialog.close)
        self.worker.start()

    def show_pred(self, result):
        self.pred = result[0]
        self.image_pred = result[1]
        self.set_image_from_cv(self.image_pred, 2)
        self.analyse_button.setEnabled(True)

    def show_detection(self):
        self.combined_image = utils.show_all_detections(self.image, self.pred)
        self.set_image_from_cv(self.combined_image, 3)

    def mallassez_calc(self):
        self.con_clean, self.con_dirty = utils.calcul_malassez(self.image, self.pred)

        self.mallassez_clean_result.setText(str(round(self.con_clean,2)) + "x10\u2075")
        self.mallassez_dirty_result.setText(str(round(self.con_dirty,2)) + "x10\u2075")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
   
    QtCore.QDir.addSearchPath('Assets', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Assets'))
    app.setWindowIcon(QtGui.QIcon("Assets:Logo_small.png"))
    file = QtCore.QFile('Assets:Style.qss')
    file.open(QtCore.QFile.ReadOnly | QtCore.QFile.Text)
    app.setStyleSheet(str(file.readAll(), 'utf-8'))
    
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()

    sys.exit(app.exec_())
