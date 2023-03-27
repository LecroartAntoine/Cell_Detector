import sys
from PyQt5.QtWidgets import QApplication, QDialog, QProgressBar, QPushButton, QVBoxLayout, QFileDialog, QInputDialog, QFormLayout, QLineEdit, QProgressDialog
from PyQt5.QtCore import Qt, QObject, pyqtSignal, pyqtSlot, QThread

class Worker(QObject):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

    @pyqtSlot()
    def work(self):
        result = 0
        for i in range(1, 10000001):
            result += i
            self.progress.emit(i)
        self.finished.emit(result)

class MyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.progress_dialog = QProgressDialog("Calculating...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowTitle("Progress")
        self.progress_dialog.setWindowModality(Qt.WindowModal)

        self.button = QPushButton("Start", self)
        self.button.clicked.connect(self.start_worker)

        layout = QVBoxLayout(self)
        layout.addWidget(self.button)

        self.worker_thread = QThread()
        self.worker = Worker()
        self.worker.moveToThread(self.worker_thread)
        self.worker.progress.connect(self.worker_progress)
        self.worker.finished.connect(self.worker_finished)
        

    def start_worker(self):
        self.worker_thread.start()
        self.worker_thread.started.connect(self.worker.work)

        self.progress_dialog.setValue(0)
        self.progress_dialog.show()
        self.button.setEnabled(False)

    @pyqtSlot(int)
    def worker_progress(self, value):
        self.progress_dialog.setValue(value)

    @pyqtSlot(object)
    def worker_finished(self, result):
        self.worker_thread.quit()
        self.worker_thread.wait()
        self.progress_dialog.hide()
        self.button.setEnabled(True)
        print("Result:", result)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    dialog = MyDialog()
    dialog.show()
    sys.exit(app.exec_())
