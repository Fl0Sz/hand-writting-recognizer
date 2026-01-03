# =========================
# eval_mnist.py
# 功能：
# 1. 載入訓練完成的 MNIST CNN 模型
# 2. 在測試資料集上評估模型準確度
# 3. 輸出整體準確率與各數字的分類準確率
# =========================

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model_infer import SimpleCNN


def main():
    """
    主程式：
    - 載入測試資料集
    - 載入模型權重
    - 計算 overall accuracy 與 per-class accuracy
    """

    # 自動選擇運算裝置（cpu / cuda）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    # 與訓練階段一致的影像前處理（Tensor + Normalize）
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 載入 MNIST 測試集（train=False）
    test_ds = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=tfm
    )

    # DataLoader：一次取多筆，加快評估速度
    test_loader = DataLoader(
        test_ds,
        batch_size=512,
        shuffle=False,
        num_workers=0
    )

    # 建立模型並載入訓練完成的權重
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
    model.eval()  # 設為推論模式

    # 整體準確度統計
    total = 0
    correct = 0

    # 各類別（0~9）的統計
    cls_total = [0] * 10
    cls_correct = [0] * 10

    # 關閉梯度計算（節省記憶體與運算）
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            # 前向推論，取機率最大的類別
            pred = model(x).argmax(dim=1)

            # 整體正確率統計
            total += y.numel()
            correct += (pred == y).sum().item()

            # 各類別正確率統計
            for i in range(y.numel()):
                label = int(y[i].item())
                cls_total[label] += 1
                if int(pred[i].item()) == label:
                    cls_correct[label] += 1

    # 計算整體準確率
    acc = correct / total
    print(f"\nOverall test accuracy: {acc:.4f} ({correct}/{total})\n")

    # 輸出各數字的分類準確率
    print("Per-class accuracy:")
    for d in range(10):
        a = cls_correct[d] / cls_total[d] if cls_total[d] else 0.0
        print(f"  Digit {d}: {a:.4f} ({cls_correct[d]}/{cls_total[d]})")


if __name__ == "__main__":
    main()
