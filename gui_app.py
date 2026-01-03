# =========================
# gui_app.py
# 功能：
# 1. 桌面 GUI（PySide6）
# 2. 提供手寫畫布：可寫一整行數字
# 3. 可上傳圖片：辨識圖片中的一行數字
# 4. 即時展示辨識結果（逐字 + 整串）
# =========================

import sys
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen
from PySide6.QtCore import Qt

from model_infer import load_model, predict_digit
from segment import segment_digits_from_bgr



def np_to_qimage_gray(gray_u8: np.ndarray) -> QImage:
    """numpy 灰階影像 -> QImage（用於顯示除錯二值圖）"""
    h, w = gray_u8.shape
    return QImage(gray_u8.data, w, h, w, QImage.Format_Grayscale8).copy()


def np_to_qimage_bgr(bgr: np.ndarray) -> QImage:
    """numpy BGR 影像 -> QImage（用於顯示預覽圖）"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


class DrawCanvas(QLabel):
    """
    手寫畫布：
    - 黑底白筆
    - 可寫一整行數字
    """
    def __init__(self, w=640, h=200):
        super().__init__()
        self.setFixedSize(w, h)

        # 畫布影像（QImage）
        self.image = QImage(w, h, QImage.Format_RGB888)
        self.image.fill(Qt.black)

        # 白色筆刷（線寬可調）
        self.pen = QPen(Qt.white, 14, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.last_pos = None

        self.setPixmap(QPixmap.fromImage(self.image))

    def clear(self):
        """清空畫布"""
        self.image.fill(Qt.black)
        self.setPixmap(QPixmap.fromImage(self.image))

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.last_pos = e.position().toPoint()

    def mouseMoveEvent(self, e):
        if (e.buttons() & Qt.LeftButton) and (self.last_pos is not None):
            p = e.position().toPoint()
            painter = QPainter(self.image)
            painter.setPen(self.pen)
            painter.drawLine(self.last_pos, p)
            painter.end()

            self.last_pos = p
            self.setPixmap(QPixmap.fromImage(self.image))

    def mouseReleaseEvent(self, e):
        self.last_pos = None

    def to_bgr(self) -> np.ndarray:
        """
        將畫布 QImage 轉成 OpenCV 可用的 BGR ndarray
        （PySide6 新版用 constBits + np.array 方式）
        """
        qimg = self.image.convertToFormat(QImage.Format_RGB888)
        w, h = qimg.width(), qimg.height()

        ptr = qimg.constBits()
        arr = np.array(ptr, copy=False).reshape(h, w, 3)  # RGB
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        return bgr.copy()


class MainWindow(QWidget):
    """
    主視窗：
    - 左：畫布
    - 右：結果 + 預覽框選 + 二值化除錯圖
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手寫數字辨識（桌面版：一行 + 上傳圖片）")

        # 載入模型（支援 .py 與 exe）
        self.model, self.device = load_model("mnist_cnn.pt")

        # UI 元件
        self.canvas = DrawCanvas(640, 200)

        self.lbl_result = QLabel("結果：")
        self.lbl_result.setStyleSheet("font-size: 18px; font-weight: 600;")

        self.lbl_preview = QLabel()
        self.lbl_preview.setFixedSize(320, 200)
        self.lbl_preview.setStyleSheet("border: 1px solid #999;")

        self.lbl_debug = QLabel()
        self.lbl_debug.setFixedSize(320, 200)
        self.lbl_debug.setStyleSheet("border: 1px solid #999;")

        # 按鈕
        btn_predict = QPushButton("辨識（畫布）")
        btn_clear = QPushButton("清除")
        btn_open = QPushButton("上傳圖片辨識")

        btn_predict.clicked.connect(self.predict_from_canvas)
        btn_clear.clicked.connect(self.canvas.clear)
        btn_open.clicked.connect(self.predict_from_image)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(QLabel("提示：寫一整行（例如 12345），字與字盡量留一點空隙。"))
        layout.addWidget(self.canvas)

        row_btn = QHBoxLayout()
        row_btn.addWidget(btn_predict)
        row_btn.addWidget(btn_clear)
        row_btn.addWidget(btn_open)
        layout.addLayout(row_btn)

        layout.addWidget(self.lbl_result)

        row_preview = QHBoxLayout()
        left_box = QVBoxLayout()
        left_box.addWidget(QLabel("預覽（框選偵測結果）"))
        left_box.addWidget(self.lbl_preview)
        right_box = QVBoxLayout()
        right_box.addWidget(QLabel("二值化除錯圖"))
        right_box.addWidget(self.lbl_debug)
        row_preview.addLayout(left_box)
        row_preview.addLayout(right_box)

        layout.addLayout(row_preview)

        self.setLayout(layout)

    def _run_pipeline(self, bgr: np.ndarray):
        digits28, boxes, bw = segment_digits_from_bgr(bgr)

        if len(digits28) == 0:
            return "", bw, bgr

        preds = []
        for d28 in digits28:
            pred, _ = predict_digit(self.model, self.device, d28)
            preds.append(str(pred))

        preview = bgr.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return "".join(preds), bw, preview

    def predict_from_canvas(self):
        """從畫布影像做辨識"""
        bgr = self.canvas.to_bgr()
        text, bw, preview = self._run_pipeline(bgr)

        if text == "":
            QMessageBox.information(self, "提示", "沒偵測到筆畫。請寫大一點、字與字分開一點。")
            return

        self.lbl_result.setText(f"結果：{text}")

        # 顯示預覽（框選後）
        qimg_prev = np_to_qimage_bgr(preview)
        self.lbl_preview.setPixmap(QPixmap.fromImage(qimg_prev).scaled(
            self.lbl_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 顯示二值化除錯圖
        qimg_bw = np_to_qimage_gray(bw)
        self.lbl_debug.setPixmap(QPixmap.fromImage(qimg_bw).scaled(
            self.lbl_debug.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def predict_from_image(self):
        """上傳圖片辨識"""
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇圖片", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not path:
            return

        bgr = cv2.imread(path)
        if bgr is None:
            QMessageBox.warning(self, "錯誤", "讀取圖片失敗。")
            return

        text, bw, preview = self._run_pipeline(bgr)

        if text == "":
            QMessageBox.information(self, "提示", "圖片中沒偵測到數字筆畫，可能太淡或背景太雜。")
            return

        self.lbl_result.setText(f"結果：{text}")

        qimg_prev = np_to_qimage_bgr(preview)
        self.lbl_preview.setPixmap(QPixmap.fromImage(qimg_prev).scaled(
            self.lbl_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        qimg_bw = np_to_qimage_gray(bw)
        self.lbl_debug.setPixmap(QPixmap.fromImage(qimg_bw).scaled(
            self.lbl_debug.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

