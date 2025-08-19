# %%
import math
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, List, Literal, Optional, Tuple, Union, cast
import random
from pathlib import Path
import pandas as pd
import numpy as np
import rtdl_our
import torch
import torch.nn as nn
import zero
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from torch import Tensor
from torch.nn import Parameter
from tqdm import trange
import lib
import json
import torch.optim as optim
import os

# %%
@dataclass
class FourierFeaturesOptions:
    n: int
    sigma: float

@dataclass
class AutoDisOptions:
    n_meta_embeddings: int
    temperature: float

@dataclass
class Config:
    @dataclass
    class Data:
        dataset: str
        type: str
        missingrate: float
        path: str
        T: lib.Transformations = field(default_factory=lib.Transformations)
        T_cache: bool = False

    @dataclass
    class Model:
        d_num_embedding: Optional[int] = None
        d_cat_embedding: Union[int, Literal['d_num_embedding'], None] = None
        num_embedding_arch: List[str] = field(default_factory=list)
        memory_efficient: bool = False
        positional_encoding: Optional[dict] = None
        periodic: Optional[dict] = None
        autodis: Optional[AutoDisOptions] = None
        k: Optional[int] = None  # embedding dimension for missing/present states
        encoder_type: str = 'transformer'  # 'transformer' or 'mlp'
        encoder_config: dict = field(default_factory=dict)

    @dataclass
    class Training:
        batch_size: int = 256
        eval_batch_size: int = 8192
        patience: int = 16
        n_epochs: int = float('inf')
        
    data: Data
    model: Model
    training: Training
    seed: int = 0

# %%
class ValueEmbedding(nn.Module):
    """数值特征嵌入模块"""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        # 使用简单的线性变换将标量值映射到embedding空间
        self.linear = nn.Linear(1, embedding_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, ) -> (B, 1) -> (B, embedding_dim)
        return self.linear(x.unsqueeze(-1))

class MissingValueEmbedding(nn.Module):
    """缺失值感知的特征嵌入模块"""
    def __init__(self, n_features: int, embedding_dim: int):
        super().__init__()
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        
        # 数值特征嵌入
        self.value_embedding = ValueEmbedding(embedding_dim)
        
        # 缺失状态和存在状态的嵌入表
        self.missing_embeddings = nn.Embedding(n_features, embedding_dim)
        self.present_embeddings = nn.Embedding(n_features, embedding_dim)
        
        # 初始化
        nn.init.normal_(self.missing_embeddings.weight, std=0.1)
        nn.init.normal_(self.present_embeddings.weight, std=0.1)
        
    def forward(self, x_hat: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x_hat: 填充后的数值特征 (B, d)
            mask: 缺失掩码，1表示缺失，0表示存在 (B, d)
        Returns:
            tokens: 特征tokens (B, d, 2*embedding_dim)
        """
        batch_size, n_features = x_hat.shape
        device = x_hat.device
        
        tokens = []
        
        for j in range(n_features):
            # 获取第j个特征的值和掩码
            feature_values = x_hat[:, j]  # (B,)
            feature_mask = mask[:, j].unsqueeze(-1)  # (B, 1)
            
            # 值嵌入
            value_emb = self.value_embedding(feature_values)  # (B, k)
            
            # 状态嵌入
            feature_idx = torch.tensor(j, device=device)
            missing_emb = self.missing_embeddings(feature_idx)  # (k,)
            present_emb = self.present_embeddings(feature_idx)  # (k,)
            
            # 根据掩码选择状态嵌入
            state_emb = (1 - feature_mask) * present_emb + feature_mask * missing_emb  # (B, k)
            
            # 组合值嵌入和状态嵌入
            # 对于缺失的特征，值嵌入被掩盖
            masked_value_emb = value_emb * (1 - feature_mask)  # (B, k)
            
            # 拼接值嵌入和状态嵌入
            token = torch.cat([masked_value_emb, state_emb], dim=-1)  # (B, 2k)
            tokens.append(token)
            
        tokens = torch.stack(tokens, dim=1)  # (B, d, 2k)
        return tokens

class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, d_model: int, n_heads: int = 8, n_layers: int = 3, 
                 d_feedforward: int = None, dropout: float = 0.1):
        super().__init__()
        if d_feedforward is None:
            d_feedforward = 4 * d_model
            
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, d, d_model)
        return self.transformer(x)

class MLPEncoder(nn.Module):
    """MLP编码器"""
    def __init__(self, input_dim: int, hidden_dim: int = None, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
            
        layers = []
        current_dim = input_dim
        
        for i in range(n_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
            
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, d, input_dim) -> (B, d, hidden_dim)
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, x.size(-1))  # (B*d, input_dim)
        output_flat = self.mlp(x_flat)  # (B*d, hidden_dim)
        return output_flat.view(batch_size, seq_len, -1)  # (B, d, hidden_dim)

class MissingAwareModel(nn.Module):
    """缺失值感知模型"""
    def __init__(self, config: Config, n_features: int, n_classes: int):
        super().__init__()
        self.config = config
        self.n_features = n_features
        self.n_classes = n_classes
        
        # 嵌入维度
        self.embedding_dim = config.model.k or 64
        
        # 缺失值embedding
        self.missing_embedding = MissingValueEmbedding(n_features, self.embedding_dim)
        
        # 编码器
        encoder_input_dim = 2 * self.embedding_dim  # 值嵌入 + 状态嵌入
        if config.model.encoder_type == 'transformer':
            self.encoder = TransformerEncoder(
                d_model=encoder_input_dim,
                n_heads=config.model.encoder_config.get('n_heads', 8),
                n_layers=config.model.encoder_config.get('n_layers', 3),
                dropout=config.model.encoder_config.get('dropout', 0.1)
            )
            encoder_output_dim = encoder_input_dim
        else:  # MLP
            hidden_dim = config.model.encoder_config.get('hidden_dim', encoder_input_dim)
            self.encoder = MLPEncoder(
                input_dim=encoder_input_dim,
                hidden_dim=hidden_dim,
                n_layers=config.model.encoder_config.get('n_layers', 2),
                dropout=config.model.encoder_config.get('dropout', 0.1)
            )
            encoder_output_dim = hidden_dim
            
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(encoder_output_dim, encoder_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(encoder_output_dim // 2, n_classes)
        )
        
    def forward(self, x_hat: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x_hat: 填充后的数值特征 (B, d)
            mask: 缺失掩码 (B, d)
        Returns:
            logits: 分类logits (B, n_classes)
        """
        # 特征嵌入
        tokens = self.missing_embedding(x_hat, mask)  # (B, d, 2k)
        
        # 编码
        encoded = self.encoder(tokens)  # (B, d, h)
        
        # 全局池化
        pooled = encoded.mean(dim=1)  # (B, h)
        
        # 分类
        logits = self.classifier(pooled)  # (B, n_classes)
        
        return logits

# %%
def evaluate(model, X_num, Y, mask, parts):
    model.eval()
    metrics = {}
    
    with torch.no_grad():
        for part in parts:
            if part not in X_num:
                continue
                
            x_num_part = X_num[part]
            y_part = Y[part]
            mask_part = mask[part]
            
            logits = model(x_num_part, mask_part)
            
            if logits.shape[1] == 1:  # 回归
                predictions = logits.squeeze(-1)
                if hasattr(y_part, 'float'):
                    y_part = y_part.float()
                loss = nn.functional.mse_loss(predictions, y_part).item()
                metrics[f'{part}_loss'] = loss
            else:  # 分类
                loss = nn.functional.cross_entropy(logits, y_part).item()
                accuracy = (logits.argmax(dim=1) == y_part).float().mean().item()
                metrics[f'{part}_loss'] = loss
                metrics[f'{part}_accuracy'] = accuracy
                
    return metrics

def main():
    config = Config(
        data=Config.Data(
            dataset='HI',
            type='mcar',
            missingrate=0.1,
            path='/root/autodl-tmp/MISS_lzy/dataset/HI/mcar_HI_0.1.csv'
        ),
        model=Config.Model(
            k=64,  # embedding维度
            encoder_type='transformer',  # 'transformer' 或 'mlp'
            encoder_config={
                'n_heads': 8,
                'n_layers': 3,
                'dropout': 0.1
            }
        ),
        training=Config.Training(
            batch_size=256,
            eval_batch_size=1024,
            patience=16,
            n_epochs=100
        ),
        seed=42
    )

    zero.improve_reproducibility(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading dataset...")
    df = pd.read_csv(config.data.path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    mask = np.isnan(X).astype(np.float32)
    
    X_filled = X.copy()
    for i in range(X.shape[1]):
        column_mean = np.nanmean(X[:, i])
        X_filled[np.isnan(X_filled[:, i]), i] = column_mean
    
    n_samples = X.shape[0]
    train_size = int(0.8 * n_samples)
    val_size = int(0.1 * n_samples)
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    X_num = {
        'train': torch.FloatTensor(X_filled[train_indices]).to(device),
        'val': torch.FloatTensor(X_filled[val_indices]).to(device),
        'test': torch.FloatTensor(X_filled[test_indices]).to(device)
    }
    
    Y = {
        'train': torch.LongTensor(y[train_indices]).to(device),
        'val': torch.LongTensor(y[val_indices]).to(device),
        'test': torch.LongTensor(y[test_indices]).to(device)
    }
    
    mask_dict = {
        'train': torch.FloatTensor(mask[train_indices]).to(device),
        'val': torch.FloatTensor(mask[val_indices]).to(device),
        'test': torch.FloatTensor(mask[test_indices]).to(device)
    }
    
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    
    print(f"Dataset info:")
    print(f"  - Features: {n_features}")
    print(f"  - Classes: {n_classes}")
    print(f"  - Train samples: {len(train_indices)}")
    print(f"  - Val samples: {len(val_indices)}")
    print(f"  - Test samples: {len(test_indices)}")
    print(f"  - Missing rate: {mask.mean():.3f}")
    
    model = MissingAwareModel(config, n_features, n_classes).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.n_epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_accuracy = 0
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(config.training.n_epochs):
        model.train()
        epoch_losses = []
        train_size = X_num['train'].shape[0]
        batch_size = config.training.batch_size
        
        for start_idx in range(0, train_size, batch_size):
            end_idx = min(start_idx + batch_size, train_size)
            
            batch_x = X_num['train'][start_idx:end_idx]
            batch_y = Y['train'][start_idx:end_idx]
            batch_mask = mask_dict['train'][start_idx:end_idx]
            
            optimizer.zero_grad()
            
            logits = model(batch_x, batch_mask)
            loss = criterion(logits, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        scheduler.step()
        
        # 评估
        if epoch % 5 == 0:
            metrics = evaluate(model, X_num, Y, mask_dict, ['train', 'val'])
            
            val_accuracy = metrics.get('val_accuracy', 0)
            
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {metrics['train_loss']:.4f} | "
                  f"Train Acc: {metrics['train_accuracy']:.4f} | "
                  f"Val Loss: {metrics['val_loss']:.4f} | "
                  f"Val Acc: {val_accuracy:.4f}")
            
            # 早停检查
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= config.training.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # 最终评估
    model.load_state_dict(torch.load('best_model.pth'))
    final_metrics = evaluate(model, X_num, Y, mask_dict, ['train', 'val', 'test'])
    
    print("\nFinal Results:")
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 清理临时文件
    if os.path.exists('best_model.pth'):
        os.remove('best_model.pth')

if __name__ == "__main__":
    main() 