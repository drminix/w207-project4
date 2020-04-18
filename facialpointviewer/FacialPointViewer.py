import sys
import logging
import os
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QMessageBox, QFileDialog
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from tensorflow.keras.models import load_model
from MainWindow import Ui_MainWindow
import numpy as np
import cv2
from collections import namedtuple

DEFAULT_MODEL_FILE = r"model/VGG16_DataAug.h5"
OPENCV_FACE_DETECTION_XML = r"model/haarcascade_frontalface_alt.xml"
TARGET_IMAGE_WIDTH = 96
TARGET_IMAGE_HEIGHT = 96
logger = logging.getLogger(__name__)
FP_COLOR_RGB = (255, 255, 0)
GLASSES_IMAGE_FILE = r"res/glasses.png"
CIGARETTE_IMAGE_FILE = r"res/cigarette.png"
N95_IMAGE_FILE = r"res/n95mask.png"
Point = namedtuple("Point", ["x", "y"])
LEFT_EYE_OUTER_CORNER = Point(6, 7)
RIGHT_EYE_OUTER_CORNER = Point(10, 11)
LEFT_EYEBROW_OUTER_END = Point(14, 15)
RIGHT_EYEBROW_OUTER_END = Point(18, 19)
MOUSE_CENTER_BOTTOM_LIP = Point(28, 29)
MOUSE_CENTER_TOP_LIP = Point(26, 27)
MOUSE_LEFT_CORNER = Point(22,23)
MOUSE_RIGHT_CORNER = Point(24,25)
NOSE_TIP = Point(20, 21)
DEFAULT_MAIN_WINDOW_WIDTH = 800
DEFAULT_MAIN_WINDOW_HEIGHT = 800


class QtCapture(QWidget):
    """Custom GUI for capturing video, performing facial point detections and displaying the results in the video"""

    def __init__(self, mainwindow, model_name, fps=60):
        super(QWidget, self).__init__()

        self._mainwindow = mainwindow
        self._fps = fps
        self.video_frame = QLabel()
        lay = QVBoxLayout()
        lay.addWidget(self.video_frame)
        self.setLayout(lay)
        self._model_name = model_name
        self._isReady = self.loadResources()
        self._display_glasses = False
        self._display_cigarette = False
        self._display_n95mask = False
        self._display_points = False
        self._reload_model = False

    def setFPS(self, fps):
        """Adjust Frames Per Second"""
        self._fps = fps

    def setModel(self, model_name):
        self._reload_model = True
        self._model_name = model_name

    def add_image(self, target_image, source_image, x, y, width, height):
        """Draws a given source image in the target image"""
        try:
            source_image = cv2.resize(source_image, (width, height))
            # copy the area we want to manipulate
            bg = target_image[y : y + height, x : x + width]
            # add the image in
            mask = source_image[:, :, 3]
            bg[mask!=0] = 0
            source_image[mask==0] = 0
            np.add(
                bg,
                source_image[:, :, 0:3],
                out=bg,
            )
            # put the changed image back into the scene
            target_image[y : y + height, x : x + width] = bg
        except:
            # continue to next frame when image cannot be fit
            return

    def nextFrameSlot(self):
        """Capture the next frame, perform facal point detections, and display it"""
        ret, frame = self.cap.read()
        # OpenCV yields frames in BGR format
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            # add in the rectangles
            cv2.rectangle(color, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # get the prediction
            cropped_face = gray[y : y + h, x : x + w]
            scale_x = cropped_face.shape[0] / TARGET_IMAGE_WIDTH
            scale_y = cropped_face.shape[1] / TARGET_IMAGE_HEIGHT
            if (
                cropped_face.shape[0] < TARGET_IMAGE_WIDTH
                or cropped_face.shape[1] < TARGET_IMAGE_HEIGHT
            ):
                continue

            cropped_face_resized = cv2.resize(
                cropped_face,
                dsize=(TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT),
                interpolation=cv2.INTER_CUBIC,
            )
            X_test = cropped_face_resized.reshape(
                1, TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT, 1
            )
            facial_point = self._model.predict(X_test)

            # plot the prediction
            scaled_fps = []
            for i in range(0, facial_point.shape[1], 2):
                center_x = int(x + facial_point[0, i] * scale_x)
                center_y = int(y + facial_point[0, i + 1] * scale_y)
                scaled_fps.extend([center_x, center_y])
                if self._display_points:
                    cv2.circle(color, (center_x, center_y), 2, FP_COLOR_RGB, 2, 8, 0)

            # display images
            # (a) glasses
            if self._display_glasses:
                # resize glasses to a new var called small glasses
                smallglasses_width = (
                    scaled_fps[LEFT_EYEBROW_OUTER_END.x]
                    - scaled_fps[RIGHT_EYEBROW_OUTER_END.x]
                    + 40
                )
                smallglasses_height = (
                    scaled_fps[RIGHT_EYE_OUTER_CORNER.y]
                    - scaled_fps[RIGHT_EYEBROW_OUTER_END.y]
                    + 20
                )
                smallglasses_y = scaled_fps[RIGHT_EYEBROW_OUTER_END.y]
                smallglasses_x = scaled_fps[RIGHT_EYEBROW_OUTER_END.x] - 20
                self.add_image(
                    color,
                    self._glasses,
                    smallglasses_x,
                    smallglasses_y,
                    smallglasses_width,
                    smallglasses_height,
                )

            # (b) cigarette
            if self._display_cigarette:
                cigarette_width = 100
                cigarette_height = (
                    scaled_fps[MOUSE_CENTER_BOTTOM_LIP.y]
                    - scaled_fps[MOUSE_CENTER_TOP_LIP.y]
                    - 60
                )
                cigarette_height = 10 if cigarette_height < 10 else cigarette_height
                cigarette_y = scaled_fps[MOUSE_CENTER_TOP_LIP.y]
                cigarette_x = scaled_fps[MOUSE_CENTER_BOTTOM_LIP.x]
                self.add_image(
                    color,
                    self._cigarette,
                    cigarette_x,
                    cigarette_y,
                    cigarette_width,
                    cigarette_height,
                )

            # (d) N95 mask
            if self._display_n95mask:
                n95mask_width = int((
                    scaled_fps[LEFT_EYEBROW_OUTER_END.x]
                    - scaled_fps[RIGHT_EYEBROW_OUTER_END.x]
                ) * 1.3)
                n95mask_height = int((
                    scaled_fps[MOUSE_CENTER_BOTTOM_LIP.y] - scaled_fps[NOSE_TIP.y]
                ) * 2.8)
                n95mask_x = int(scaled_fps[RIGHT_EYEBROW_OUTER_END.x] - (scaled_fps[MOUSE_LEFT_CORNER.x] - scaled_fps[MOUSE_RIGHT_CORNER.x])/3)
                n95mask_y = int(scaled_fps[NOSE_TIP.y] - (scaled_fps[MOUSE_CENTER_BOTTOM_LIP.y] - scaled_fps[MOUSE_CENTER_TOP_LIP.y]))
                self.add_image(
                    color,
                    self._n95mask,
                    n95mask_x,
                    n95mask_y,
                    n95mask_width,
                    n95mask_height,
                )

        # display the image in QT pixmap
        img = QImage(
            color, color.shape[1], color.shape[0], QImage.Format_RGB888
        ).scaled(640, 480, Qt.KeepAspectRatio)
        pix = QPixmap.fromImage(img)
        self.video_frame.setPixmap(pix)

    def start(self):
        """Start capturing data by setting up timer"""
        if self._reload_model:
            self.loadResources()
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1000.0 / self._fps)

    def stop(self):
        """Stop capturing data """
        self.cap.release()
        self.timer.stop()

    def deleteLater(self):
        self.cap.release()
        super(QWidget, self).deleteLater()

    def loadResources(self):
        """Load models & other resources"""
        # face detection filter
        self.face_cascade = cv2.CascadeClassifier(OPENCV_FACE_DETECTION_XML)
        # facial point detection model
        self._model = load_model(self._model_name)
        logger.info(f"Loaded model: {self._model_name}")
        self._mainwindow.statusBar().showMessage(f"Loaded model: {self._model_name}")
        # fashion items
        self._glasses = cv2.imread(GLASSES_IMAGE_FILE, cv2.IMREAD_UNCHANGED)
        self._cigarette = cv2.imread(CIGARETTE_IMAGE_FILE, cv2.IMREAD_UNCHANGED)
        self._n95mask = cv2.imread(N95_IMAGE_FILE, cv2.IMREAD_UNCHANGED)
        self._reload_model = False
        return True

    def setGlassesState(self, c_state):
        """Display glasses"""
        self._display_glasses = bool(c_state)

    def setCigaretteState(self, c_state):
        """Display cigarette"""
        self._display_cigarette = bool(c_state)

    def setN95maskState(self, c_state):
        self._display_n95mask = bool(c_state)

    def setPointState(self, c_state):
        """Display facial points"""
        self._display_points = bool(c_state)


class FacialPointViewer(QtWidgets.QMainWindow):
    """MainWindow class"""

    def __init__(self):
        super(FacialPointViewer, self).__init__()
        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
        self.setupUI()

    def setupUI(self):
        """setup UI"""
        logger.info(f"Loaded UI..")
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        self.setWindowTitle("Dr. Tensorflow Facial Points Detector")
        self._capture_widget = QtCapture(mainwindow=self, model_name=DEFAULT_MODEL_FILE)
        self._ui.verticalLayout.addChildWidget(self._capture_widget)
        self.setFixedSize(DEFAULT_MAIN_WINDOW_WIDTH, DEFAULT_MAIN_WINDOW_HEIGHT)
        self._ui.textEdit_filename.setText(
            os.path.join(os.getcwd(), DEFAULT_MODEL_FILE)
        )
        self._ui.menubar.setVisible(True)

        # link events
        self._ui.pushButton_StartCapture.clicked.connect(self._capture_widget.start)
        self._ui.pushButton_StopCapture.clicked.connect(self._capture_widget.stop)
        self._ui.checkBox_glasses.stateChanged.connect(
            self._capture_widget.setGlassesState
        )
        self._ui.checkBox_cigarette.stateChanged.connect(
            self._capture_widget.setCigaretteState
        )
        self._ui.checkBox_n95mask.stateChanged.connect(
            self._capture_widget.setN95maskState
        )
        self._ui.checkBox_fps.stateChanged.connect(self._capture_widget.setPointState)
        self._ui.menu_File.triggered.connect(self.closeEvent)
        self._ui.menu_About.triggered.connect(self.showAboutDialog)
        self._ui.pushButton_browse.clicked.connect(self.browseModel)

        # initial state
        self._ui.checkBox_fps.setChecked(True)

    def showAboutDialog(self):
        """show dialog"""
        msgBox = QMessageBox.about(
            self,
            "About Dr. Tensorflow Facial Points Detector",
            "Facial Points detection using CNN model",
        )

    def browseModel(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Select a model", "", "All Files (*);;Model  (*.hd5)",
        )
        if fileName:
            self._ui.textEdit_filename.setText(fileName)
            self._capture_widget.setModel(fileName)
            self._capture_widget.stop()
            self._capture_widget.start()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    facialviewer = FacialPointViewer()
    facialviewer.show()
    sys.exit(app.exec())
