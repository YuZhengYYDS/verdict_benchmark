import argparse
import yaml
import torch
import numpy as np
import os
from tqdm import tqdm
from importlib import import_module
import wandb

from data.dataset import get_dataloaders
from utils.metrics import mse_loss

# Official Train Script Supporting Advanced LR Scheduling
# Usage: python train.py --config configs/your_model.yaml

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_scheduler(optimizer, scheduler_cfg):
    """
    构建学习率调度器，支持 Warmup + CosineAnnealingWarmRestarts
    或其他 PyTorch 原生调度器。
    """
    warmup_epochs = scheduler_cfg.get('warmup_epochs', 0)
    sched_type = scheduler_cfg['type']

    # Cosine Annealing with Warm Restarts
    if sched_type == 'CosineAnnealingWarmRestarts':
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_cfg.get('T_0', 10),
            T_mult=scheduler_cfg.get('T_mult', 1),
            eta_min=scheduler_cfg.get('eta_min', 0.0)
        )
    else:
        # 其他调度器类型
        cls = getattr(torch.optim.lr_scheduler, sched_type)
        # 排除 type 和 warmup_epochs
        args = {k: v for k, v in scheduler_cfg.items() if k not in ['type', 'warmup_epochs']}
        scheduler = cls(optimizer, **args)

    return scheduler, warmup_epochs


def main(config_path):
    # 1. 读取配置
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 2. 初始化 W&B
    wandb.init(
        project=cfg.get('wandb_project', 'verdict_benchmark'),
        name=cfg.get('wandb_run_name'),
        config=cfg
    )

    print(f"[INFO] Loaded config: {config_path}")
    set_seed(cfg['seed'])

    # 3. 数据加载
    print("[INFO] Loading data...")
    train_loader, val_loader = get_dataloaders(
        mat_path=cfg['data']['mat_path'],
        batch_size=cfg['batch_size'],
        train_ratio=cfg['train_ratio'],
        seed=cfg['seed'],
        scaler_path=os.path.join('checkpoints', f"{cfg['model']['type']}_scaler.pkl")
    )
    print("[INFO] Data ready.")

    # 4. 模型实例化
    module = import_module(f"models.{cfg['model']['type']}")
    Model = getattr(module, cfg['model']['class_name'])
    # 样本维度
    X0, y0 = next(iter(train_loader))
    model = Model(input_dim=X0.shape[1], output_dim=y0.shape[1], **cfg['model']['params'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg.get('weight_decay', 0))
    criterion = torch.nn.MSELoss()

    # 5. Scheduler 设置
    scheduler, warmup_epochs = build_scheduler(optimizer, cfg.get('scheduler', {}))
    base_lr = cfg['lr']

    # 6. 目录创建
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    best_loss = float('inf')
    patience = 0

    # 7. 训练循环
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        train_loss = 0.0

        # Warmup LR
        if epoch <= warmup_epochs:
            lr = base_lr * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        else:
            # 使用调度器更新 LR
            if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                scheduler.step(epoch - 1)
            else:
                scheduler.step()
            lr = optimizer.param_groups[0]['lr']

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            model_output = model(X)
            
            # Handle models that return tuples (e.g., VAE, MoE, Ensemble, Hybrid)
            if isinstance(model_output, tuple):
                pred = model_output[0]  # First element is always the prediction
                # For models like MoE, second element might be diagnostic info, not loss
                # Only treat as auxiliary loss if it's a scalar tensor
                aux_loss = 0.0
                if len(model_output) > 1 and model_output[1] is not None:
                    potential_aux_loss = model_output[1]
                    # Check if it's a scalar loss (0-dimensional tensor) or can be reduced to scalar
                    if isinstance(potential_aux_loss, torch.Tensor):
                        if potential_aux_loss.dim() == 0:  # Already scalar
                            aux_loss = potential_aux_loss
                        elif potential_aux_loss.numel() == 1:  # Single element tensor
                            aux_loss = potential_aux_loss.item()
                        # Otherwise, it's diagnostic info (like gating weights), not a loss
            else:
                pred = model_output
                aux_loss = 0.0
                
            # Primary loss (prediction)
            primary_loss = criterion(pred, y)
            
            # Total loss (primary + auxiliary)
            total_loss = primary_loss + aux_loss
            
            total_loss.backward()
            optimizer.step()
            train_loss += primary_loss.item() * X.size(0)

        train_loss /= len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                model_output = model(X)
                
                # Handle models that return tuples (e.g., VAE, MoE, Ensemble, Hybrid)
                if isinstance(model_output, tuple):
                    pred = model_output[0]  # First element is always the prediction
                else:
                    pred = model_output
                    
                val_loss += criterion(pred, y).item() * X.size(0)
        val_loss /= len(val_loader.dataset)

        # W&B 记录
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'lr': lr})
        print(f"Epoch {epoch}/{cfg['epochs']} | LR: {lr:.6e} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

                # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            save_path = os.path.join('checkpoints', f"{cfg['model']['type']}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] Saved best model to {save_path}")
        else:
            patience += 1
            if patience >= cfg['early_stop_patience']:
                print("Early stopping triggered.")
                break

    # 结束 W&B 记录
    wandb.finish()
    print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args.config)
