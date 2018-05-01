import sys
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QApplication, QLabel
from PyQt5.QtGui import QPixmap, QImage, QFont
from ifastpcp import *


class Thread(QThread):
    changePixmapLr = pyqtSignal(QPixmap)
    changePixmapSp = pyqtSignal(QPixmap)

    def __init__(self, parent=None):
        QThread.__init__(self, parent=parent)

    def run(self):
        cap = cv2.VideoCapture(0)

        # Init for PCP
        scale = 0.2
        k0 = 3

        init_frames = []
        for i in range(k0):
            ret, frame = cap.read()        
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            init_frames.append(gray)

        h,w = init_frames[0].shape
        rh = int(h * scale)
        rw = int(w * scale)
        ims = [cv2.resize(gray, (rw, rh)) for gray in init_frames]
        vecs = [np.atleast_2d(im.flatten()).T for im in ims]
        init = np.hstack(vecs)
        lm = 1 / np.sqrt(rh*rw)
        r = 1
        i = k0

        uk, sk, vk = incPCP_init(init, r)
        vk = vk.T


        while True:
            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            im = cv2.resize(gray, (rw, rh))
            d = im.flatten()
            
            Lk, Sk, uk, sk, vk = incPCP_update(d, uk, sk, vk, lm, r, i)
            vk = vk.T

            cv2.normalize(Lk, Lk, 0, 255, cv2.NORM_MINMAX)
            cv2.normalize(Sk, Sk, 0, 255, cv2.NORM_MINMAX)
            Lk = Lk.astype(np.uint8)
            Sk = Sk.astype(np.uint8)
            lr = Lk.reshape(rh,rw)
            sp = Sk.reshape(rh,rw)
            
            convertToQtFormat = QImage(lr.data, lr.shape[1], lr.shape[0], QImage.Format_Grayscale8)
            convertToQtFormat = QPixmap.fromImage(convertToQtFormat)
            p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
            self.changePixmapLr.emit(p)

            convertToQtFormat = QImage(sp.data, sp.shape[1], sp.shape[0], QImage.Format_Grayscale8)
            convertToQtFormat = QPixmap.fromImage(convertToQtFormat)
            p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
            self.changePixmapSp.emit(p)
            
            i += 1



class App(QWidget):
    def __init__(self):
        super(App, self).__init__()
        self.title = 'PyQt5 Video'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(1800, 1200)

        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(36)

        label_lr_title = QLabel(self)
        label_lr_title.setText("Low-rank")
        label_lr_title.setFont(font)
        label_lr_title.move(360, 100)
        label_lr = QLabel(self)
        label_lr.move(40, 120)
        label_lr.resize(640, 480)

        label_sp_title = QLabel(self)
        label_sp_title.setText("Sparse")
        label_sp_title.setFont(font)
        label_sp_title.move(1080, 100)
        label_sp = QLabel(self)
        label_sp.move(760, 120)
        label_sp.resize(640, 480)

        th = Thread(self)
        th.changePixmapLr.connect(label_lr.setPixmap)
        th.changePixmapSp.connect(label_sp.setPixmap)
        th.start()
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
