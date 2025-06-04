import argparse
import yaml
import torch
import numpy as np
import os
from tqdm import tqdm   # 新增

from data.dataset import get_dataloaders
from models.mlp import MLPRegressor
from utils.metrics import mse_loss

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def main(config_path):
    # 1. 读取配置
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("[INFO] 配置读取完成。")
    set_seed(cfg['seed'])
    scaler_path = "checkpoints/mlp_minmaxscaler.pkl"  # 推荐也放到checkpoints
    print("[INFO] 加载数据集...")
    train_loader, val_loader = get_dataloaders(
        mat_path='D:/AiProjects/UCLmaster/dlfitting_verdict/VERDICT_training/AS_Z_fixdv/TrainingSet.mat',
        batch_size=cfg['batch_size'],
        train_ratio=cfg['train_ratio'],
        seed=cfg['seed'],
        scaler_path=scaler_path
    )
    print("[INFO] 数据加载完毕。")

    X_sample, y_sample = next(iter(train_loader))
    input_dim = X_sample.shape[1]
    output_dim = y_sample.shape[1]
    model = MLPRegressor(input_dim, output_dim, cfg['hidden_dims'], cfg['activation']).to('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    criterion = torch.nn.MSELoss()
    print("[INFO] 模型初始化完成。")

    best_loss = float('inf')
    patience = 0

    # 创建logs和checkpoints目录
    log_dir = 'logs'
    ckpt_dir = 'checkpoints'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'mlp_log.txt')
    ckpt_path = os.path.join(ckpt_dir, 'mlp_best.pt')

    # 写表头
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write("Epoch\tTrainLoss\tValLoss\n")

    print("[INFO] 开始训练...")
    for epoch in tqdm(range(cfg['epochs']), desc="[Epoch]", ncols=80):
        model.train()
        total_loss = 0
        # batch级别进度条
        for X, y in tqdm(train_loader, desc=f"Train E{epoch+1}", leave=False, ncols=70):
            X, y = X.to(model.model[0].weight.device), y.to(model.model[0].weight.device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
        avg_train_loss = total_loss / len(train_loader.dataset)

        # 验证
        model.eval()
        total_val_loss = 0
        for X, y in tqdm(val_loader, desc=f"Valid E{epoch+1}", leave=False, ncols=70):
            X, y = X.to(model.model[0].weight.device), y.to(model.model[0].weight.device)
            pred = model(X)
            loss = criterion(pred, y)
            total_val_loss += loss.item() * X.size(0)
        avg_val_loss = total_val_loss / len(val_loader.dataset)

        # 输出到控制台
        tqdm.write(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # 追加日志到logs/mlp_log.txt
        with open(log_path, 'a') as f:
            f.write(f"{epoch+1}\t{avg_train_loss:.6f}\t{avg_val_loss:.6f}\n")

        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience = 0
            torch.save(model.state_dict(), ckpt_path)
            tqdm.write(f"[INFO] 模型已保存: {ckpt_path}")
        else:
            patience += 1
            if patience >= cfg['early_stop_patience']:
                tqdm.write("[INFO] Early stopping triggered.")
                break

    print("[INFO] 训练结束。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/mlp.yaml')
    args = parser.parse_args()
    main(args.config)
