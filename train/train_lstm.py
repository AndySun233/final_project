import os
import torch
from losses.loss import nll_only  # 可以替换
from models.lstm_model import LSTMStudentT

def train_model(train_loader, input_dim,
                epochs=5, lr=1e-4,
                model_path="lstm/best.pt", loss_fn=nll_only, model=None):
    """
    LSTM 训练函数，不使用验证集，按最小 train loss 保存模型
    """
    if model is None:
        model = LSTMStudentT(input_dim=input_dim)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_train_loss = float("inf")
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            mu, sigma, nu = model(X)
            loss = loss_fn(mu, sigma, y, nu)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
        train_loss /= len(train_loader.dataset)

        is_best = train_loss < best_train_loss
        if is_best:
            best_train_loss = train_loss
            best_model_state = model.state_dict()

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.6f}" + (" <-- best" if is_best else ""))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(best_model_state, model_path)
    print(f"✅ 最佳模型已保存到 {model_path}")

    model.load_state_dict(torch.load(model_path))
    return model
