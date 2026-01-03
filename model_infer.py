# =========================
# model_infer.py
# 功能：
# 1. 定義 MNIST 手寫數字 CNN 模型
# 2. 載入訓練完成的模型權重（支援 .py 與 exe）
# 3. 提供單一數字的推論函式
# =========================

import torch
import torch.nn as nn
import numpy as np
import os
import sys


class SimpleCNN(nn.Module):
    """
    MNIST 用的簡易 CNN 分類模型
    輸入：1 x 28 x 28（灰階影像）
    輸出：10 維 logits（數字 0~9）
    """
    def __init__(self):
        super().__init__()

        # 使用 Sequential 簡化模型結構定義
        self.net = nn.Sequential(
            # 第 1 層卷積：1 -> 16 個 feature maps
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            # 下採樣，尺寸從 28x28 -> 14x14
            nn.MaxPool2d(2),

            # 第 2 層卷積：16 -> 32 個 feature maps
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            # 再次下採樣，14x14 -> 7x7
            nn.MaxPool2d(2),

            # 展平成一維向量
            nn.Flatten(),

            # 全連接層
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),

            # 輸出層：10 類（0~9）
            nn.Linear(128, 10)
        )

    def forward(self, x):
        """
        前向傳播
        """
        return self.net(x)


def _resource_path(relative_path: str) -> str:
    """
    取得資源檔案的正確路徑

    - 直接執行 .py 時：使用目前檔案所在資料夾
    - 使用 PyInstaller 打包成 exe 時：使用 sys._MEIPASS 暫存路徑

    這個設計是為了讓「原始碼執行」與「exe 執行」都能正確讀到模型檔
    """
    if hasattr(sys, "_MEIPASS"):
        # exe 執行時
        return os.path.join(sys._MEIPASS, relative_path)

    # 一般 python 執行時
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


def load_model(model_name="mnist_cnn.pt", device=None):
    """
    載入已訓練完成的模型

    參數：
    - model_name：模型權重檔名
    - device：cpu 或 cuda（預設自動判斷）

    回傳：
    - model：已載入權重並設定為 eval 模式的模型
    - device：實際使用的裝置
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 取得模型檔正確路徑（支援 exe）
    model_path = _resource_path(model_name)

    # 建立模型並載入權重
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 設定為推論模式（關閉 dropout / batchnorm 訓練行為）
    model.eval()
    return model, device


def mnist_normalize(x01: np.ndarray) -> np.ndarray:
    """
    MNIST 標準化處理
    將影像正規化為 (x - mean) / std

    這裡的 mean 與 std 來自 MNIST 官方統計值
    """
    return (x01 - 0.1307) / 0.3081


@torch.no_grad()
def predict_digit(model, device, img28_x01: np.ndarray):
    """
    單一數字推論函式

    參數：
    - model：已載入的 CNN 模型
    - device：cpu / cuda
    - img28_x01：28x28，值域 [0,1] 的影像（numpy array）

    回傳：
    - pred：預測的數字（0~9）
    - probs：10 類別的機率分佈
    """

    # 正規化（與訓練時一致）
    x = mnist_normalize(img28_x01).astype(np.float32)

    # 轉成 PyTorch tensor，並補 batch 與 channel 維度
    # shape: (1, 1, 28, 28)
    x = torch.tensor(x).unsqueeze(0).unsqueeze(0).to(device)

    # 前向推論
    logits = model(x)

    # softmax 轉成機率
    probs = torch.softmax(logits, dim=1).cpu().numpy().reshape(-1)

    # 取最大機率的類別
    pred = int(probs.argmax())

    return pred, probs
