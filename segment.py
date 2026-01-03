# =========================
# segment.py (robust)
# 功能：
# 1) 輸入影像 -> 穩定二值化（自動反相）
# 2) 連通元件找數字框（適用字有間距）
# 3) 若框太少（黏在一起） -> 使用垂直投影切割（備援）
# 4) ROI -> 28x28 MNIST 格式
# =========================

import cv2
import numpy as np


def binarize_for_digits(bgr_img: np.ndarray) -> np.ndarray:
    """回傳 bw：白字(255)黑底(0)"""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 依平均亮度自動決定要不要反相
    # 亮背景（上傳紙張）通常需要 INV；黑背景（畫布）則不一定
    mean = float(gray.mean())
    if mean > 127:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 讓筆畫更連續：先 close 再 open（先補洞再去雜）
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)

    # 輕微膨脹，避免細的 1 被斷掉
    bw = cv2.dilate(bw, np.ones((2, 2), np.uint8), iterations=1)

    return bw


def extract_digit_rois(bw: np.ndarray, min_area=60) -> list:
    """連通元件找 box，回傳由左到右排序的 (x,y,w,h)"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    boxes = []
    H, W = bw.shape
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        # 過濾雜訊
        if area < min_area:
            continue

        # 過濾太扁太長的雜訊（例如底線）
        if h < 8 or w < 3:
            continue
        if w > 0.9 * W and h < 0.3 * H:
            continue

        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])
    return boxes


def _split_by_vertical_projection(bw: np.ndarray, min_gap=3, min_width=5):
    """
    備援切割：當字黏一起時，用垂直投影找「空白縫」切開
    回傳 boxes（左到右）
    """
    H, W = bw.shape
    col_sum = (bw > 0).sum(axis=0)  # 每一列白點數

    # 找出投影接近 0 的區域當作 gap
    is_gap = col_sum <= 1
    gaps = []
    i = 0
    while i < W:
        if is_gap[i]:
            j = i
            while j < W and is_gap[j]:
                j += 1
            if (j - i) >= min_gap:
                gaps.append((i, j))
            i = j
        else:
            i += 1

    # 用 gaps 產生切割區間
    cuts = [0]
    for a, b in gaps:
        mid = (a + b) // 2
        cuts.append(mid)
    cuts.append(W)

    # 合成 boxes（每段切出一個 ROI）
    boxes = []
    for k in range(len(cuts) - 1):
        x0, x1 = cuts[k], cuts[k + 1]
        if (x1 - x0) < min_width:
            continue

        roi = bw[:, x0:x1]
        ys, xs = np.where(roi > 0)
        if len(xs) == 0:
            continue

        x = x0 + int(xs.min())
        y = int(ys.min())
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])
    return boxes


def roi_to_mnist(bw: np.ndarray, box) -> np.ndarray:
    """box 區域轉成 28x28，回傳 float [0,1]"""
    x, y, w, h = box
    roi = bw[y:y + h, x:x + w]

    # 補成正方形並置中
    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)
    x_off = (size - w) // 2
    y_off = (size - h) // 2
    square[y_off:y_off + h, x_off:x_off + w] = roi

    # 先縮到 20x20，再補邊到 28x28（更像 MNIST）
    img20 = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
    img28 = cv2.copyMakeBorder(img20, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)

    img28 = img28.astype(np.float32) / 255.0
    return img28


def segment_digits_from_bgr(bgr_img: np.ndarray):
    """
    回傳：
    digits28: list of 28x28 float [0,1]
    boxes: 對應框
    bw: 二值化除錯圖
    """
    bw = binarize_for_digits(bgr_img)
    boxes = extract_digit_rois(bw)

    # 如果偵測到的 box 太少（例如整行被當成一塊），用投影備援切割
    if len(boxes) <= 1:
        boxes2 = _split_by_vertical_projection(bw)
        if len(boxes2) > len(boxes):
            boxes = boxes2

    digits28 = [roi_to_mnist(bw, b) for b in boxes]
    return digits28, boxes, bw

