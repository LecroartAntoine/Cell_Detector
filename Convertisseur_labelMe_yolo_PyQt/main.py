from PyQt5 import QtCore, QtGui, QtWidgets
import os, json

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(600, 250)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.formLayout = QtWidgets.QFormLayout(self.centralwidget)
        self.formLayout.setObjectName("formLayout")
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setObjectName("listWidget")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.listWidget)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.open)
        self.verticalLayout.addWidget(self.pushButton)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem)

        self.widthLayout = QtWidgets.QHBoxLayout()
        self.widthLayout.setObjectName("widthLayout")
        self.width_label = QtWidgets.QLabel(self.centralwidget)
        self.width_label.setObjectName("width_label")
        self.widthLayout.addWidget(self.width_label)
        self.width = QtWidgets.QSpinBox(self.centralwidget)
        self.width.setMaximum(10000)
        self.width.setSuffix(' pixels')
        self.width.setAccelerated(True)
        self.width.setProperty("value", 2592)
        self.width.setObjectName("width")
        self.widthLayout.addWidget(self.width)
        self.verticalLayout.addLayout(self.widthLayout)

        self.heightLayout = QtWidgets.QHBoxLayout()
        self.heightLayout.setObjectName("heightLayout")
        self.height_label = QtWidgets.QLabel(self.centralwidget)
        self.height_label.setObjectName("height_label")
        self.heightLayout.addWidget(self.height_label)
        self.height = QtWidgets.QSpinBox(self.centralwidget)
        self.height.setMaximum(10000)
        self.height.setSuffix(' pixels')
        self.height.setAccelerated(True)
        self.height.setProperty("value", 1944)
        self.height.setObjectName("height")
        self.heightLayout.addWidget(self.height)
        self.verticalLayout.addLayout(self.heightLayout)

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.convert)
        self.verticalLayout.addWidget(self.pushButton_2)
        self.formLayout.setLayout(3, QtWidgets.QFormLayout.FieldRole, self.verticalLayout)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.SpanningRole, self.label_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 518, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "LabelMe vers Yolo"))
        self.pushButton.setText(_translate("MainWindow", "Ouvrir le dossier"))
        self.width_label.setText(_translate("MainWindow", "Largeur des images"))
        self.height_label.setText(_translate("MainWindow", "Hauteur des images"))
        self.pushButton_2.setText(_translate("MainWindow", "Convertir"))
        self.label_2.setText(_translate("MainWindow", "Convertisseur format LabelMe vers format Yolo"))

    def open(self):
        self.listWidget.clear()
        self.folder = str(QtWidgets.QFileDialog.getExistingDirectory(MainWindow, "Sélectionner un dossier"))
        if self.folder:
            for file in os.listdir(self.folder):
                if file.lower().endswith('.json'):
                    self.listWidget.addItem(file)

    def convert(self):
        if not os.path.exists(self.folder + '/' + 'Yolo_labels'):
            os.mkdir(self.folder + '/' + 'Yolo_labels')

        for file in os.listdir(self.folder):
            if file[-4:] == 'json':
                with open (self.folder + '/' + file) as f:
                    data = json.load(f)

                with open(f'{self.folder}/Yolo_labels/{file[:-5]}.txt', 'w') as f:
                    pass

                for i, shape in enumerate(data['shapes']):
                    x_min = shape["points"][0][0]
                    y_min = shape["points"][0][1]
                    x_max = shape["points"][1][0]
                    y_max = shape["points"][1][1]

                    x = (x_min + ((x_max - x_min) / 2)) / self.width.value()
                    y = (y_min + ((y_max - y_min) / 2)) / self.height.value()
                    w = (x_max - x_min) / self.width.value()
                    h = (y_max - y_min) / self.height.value()

                    if i != 0:
                        line = f"\n0 {x} {y} {w} {h}"
                    else:
                        line = f"0 {x} {y} {w} {h}"
                    with open(f'{self.folder}/Yolo_labels/{file[:-5]}.txt', 'a') as f:
                        f.write(line)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()

    QtCore.QDir.addSearchPath('Assets', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Assets'))
    app.setWindowIcon(QtGui.QIcon("Assets:Logo_small.png"))
    file = QtCore.QFile('Assets:Style.qss')
    file.open(QtCore.QFile.ReadOnly | QtCore.QFile.Text)
    app.setStyleSheet(str(file.readAll(), 'utf-8'))

    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
