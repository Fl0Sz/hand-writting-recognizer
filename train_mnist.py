import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model_infer import SimpleCNN  # 直接重用同一個模型定義

def main():
    # 自動選擇裝置（你目前多半是 cpu）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    # MNIST 標準 normalize（與推論一致）
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 下載/讀取資料集
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    model = SimpleCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(3):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        # 每個 epoch 後做一次測試集準確率（方便展示）
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()

        acc = correct / total
        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, test_acc={acc:.4f}")

    # 存模型權重
    torch.save(model.state_dict(), "mnist_cnn.pt")
    print("Saved: mnist_cnn.pt")

if __name__ == "__main__":
    main()
