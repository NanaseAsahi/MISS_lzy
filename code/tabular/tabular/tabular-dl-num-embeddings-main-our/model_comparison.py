#!/usr/bin/env python3
"""
模型性能对比: MLP vs XGBoost vs Logistic Regression
对比三种模型在不同特征数量下的性能表现
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

class SimpleMLP(nn.Module):
    """简单的MLP模型"""
    def __init__(self, input_dim, num_classes=2, hidden_dims=None):
        super().__init__()
        
        if hidden_dims is None:
            h1 = max(16, min(128, input_dim * 8))
            h2 = max(8, min(64, input_dim * 4))
            hidden_dims = [h1, h2]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class ModelComparator:
    """模型性能对比器"""
    
    def __init__(self, data_path, random_seed=42):
        self.data_path = data_path
        self.random_seed = random_seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置随机种子
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        
        print(f"🔧 使用设备: {self.device}")
        
    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print(f"📂 正在加载数据: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        # 分离特征和标签
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        print(f"📊 数据集形状: {df.shape}")
        print(f"📈 特征数量: {X.shape[1]}")
        print(f"📋 样本数量: {X.shape[0]}")
        print(f"🎯 标签分布: {dict(y.value_counts())}")
        
        # 处理缺失值
        if X.isnull().any().any():
            print("🔧 检测到缺失值，使用均值填充...")
            X = X.fillna(X.mean())
        
        # 标签编码
        if y.dtype == 'object' or len(y.unique()) <= 20:
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            self.num_classes = len(self.label_encoder.classes_)
            print(f"🏷️  标签已编码，类别数: {self.num_classes}")
        else:
            self.label_encoder = None
            self.num_classes = len(y.unique())
        
        # 数据划分 (60% 训练, 20% 验证, 20% 测试)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=self.random_seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.random_seed, stratify=y_temp
        )
        
        print(f"📊 数据划分 - 训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")
        
        # 特征标准化
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 存储数据
        self.X_train = X_train_scaled
        self.X_val = X_val_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        self.feature_names = list(X.columns)
        self.total_features = len(self.feature_names)
        
        print(f"✅ 数据预处理完成，总特征数: {self.total_features}")
        
    def select_top_features(self, num_features):
        """选择前n个最重要的特征（基于方差）"""
        feature_variances = np.var(self.X_train, axis=0)
        top_indices = np.argsort(feature_variances)[::-1][:num_features]
        return sorted(top_indices)
    
    def train_mlp(self, feature_indices, max_epochs=100, patience=15):
        """训练MLP模型"""
        num_features = len(feature_indices)
        
        # 准备数据
        X_train_subset = torch.FloatTensor(self.X_train[:, feature_indices]).to(self.device)
        X_val_subset = torch.FloatTensor(self.X_val[:, feature_indices]).to(self.device)
        X_test_subset = torch.FloatTensor(self.X_test[:, feature_indices]).to(self.device)
        y_train_tensor = torch.LongTensor(self.y_train).to(self.device)
        y_val_tensor = torch.LongTensor(self.y_val).to(self.device)
        y_test_tensor = torch.LongTensor(self.y_test).to(self.device)
        
        # 创建模型
        model = SimpleMLP(num_features, self.num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # 训练
        best_val_acc = 0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(max_epochs):
            # 训练模式
            model.train()
            optimizer.zero_grad()
            train_outputs = model(X_train_subset)
            train_loss = criterion(train_outputs, y_train_tensor)
            train_loss.backward()
            optimizer.step()
            
            # 验证模式
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_subset)
                val_predictions = torch.argmax(val_outputs, dim=1)
                val_acc = (val_predictions == y_val_tensor).float().mean().item()
            
            # 早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # 加载最佳模型进行最终评估
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        model.eval()
        with torch.no_grad():
            # 验证集评估
            val_outputs = model(X_val_subset)
            val_predictions = torch.argmax(val_outputs, dim=1)
            val_acc = (val_predictions == y_val_tensor).float().mean().item()
            
            # 测试集评估
            test_outputs = model(X_test_subset)
            test_predictions = torch.argmax(test_outputs, dim=1)
            test_acc = (test_predictions == y_test_tensor).float().mean().item()
            
            # 计算AUC (仅二分类)
            if self.num_classes == 2:
                val_probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
                test_probs = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
                val_auc = roc_auc_score(self.y_val, val_probs)
                test_auc = roc_auc_score(self.y_test, test_probs)
            else:
                val_auc = 0.0
                test_auc = 0.0
            
            # 计算F1分数
            val_f1 = f1_score(self.y_val, val_predictions.cpu().numpy(), average='weighted')
            test_f1 = f1_score(self.y_test, test_predictions.cpu().numpy(), average='weighted')
        
        return {
            'val_acc': val_acc,
            'val_auc': val_auc,
            'val_f1': val_f1,
            'test_acc': test_acc,
            'test_auc': test_auc,
            'test_f1': test_f1,
            'epochs_trained': epoch + 1
        }
    
    def train_xgboost(self, feature_indices):
        """训练XGBoost模型 (使用与IPS.py相同的配置)"""
        # 准备数据
        X_train_subset = self.X_train[:, feature_indices]
        X_val_subset = self.X_val[:, feature_indices]
        X_test_subset = self.X_test[:, feature_indices]
        
        # 计算类别权重平衡
        if self.num_classes == 2:
            scale_pos_weight = sum(self.y_train == 0) / sum(self.y_train == 1)
        else:
            scale_pos_weight = 1
        
        # 创建XGBoost分类器 (与IPS.py相同配置)
        xgb_model = XGBClassifier(
            objective='reg:logistic' if self.num_classes == 2 else 'multi:softprob',
            booster='gbtree',
            max_depth=3,
            verbosity=0,  # 设为0以避免输出干扰
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_seed,
            eval_metric='logloss' if self.num_classes == 2 else 'mlogloss'
        )
        
        # 训练模型
        xgb_model.fit(
            X_train_subset, self.y_train,
            eval_set=[(X_val_subset, self.y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        # 预测
        val_predictions = xgb_model.predict(X_val_subset)
        test_predictions = xgb_model.predict(X_test_subset)
        
        val_acc = accuracy_score(self.y_val, val_predictions)
        test_acc = accuracy_score(self.y_test, test_predictions)
        
        # 计算AUC (仅二分类)
        if self.num_classes == 2:
            val_probs = xgb_model.predict_proba(X_val_subset)[:, 1]
            test_probs = xgb_model.predict_proba(X_test_subset)[:, 1]
            val_auc = roc_auc_score(self.y_val, val_probs)
            test_auc = roc_auc_score(self.y_test, test_probs)
        else:
            val_auc = 0.0
            test_auc = 0.0
        
        # 计算F1分数
        val_f1 = f1_score(self.y_val, val_predictions, average='weighted')
        test_f1 = f1_score(self.y_test, test_predictions, average='weighted')
        
        return {
            'val_acc': val_acc,
            'val_auc': val_auc,
            'val_f1': val_f1,
            'test_acc': test_acc,
            'test_auc': test_auc,
            'test_f1': test_f1
        }
    
    def train_logistic_regression(self, feature_indices):
        """训练Logistic Regression模型 (使用与IPS.py相似的配置)"""
        # 准备数据
        X_train_subset = self.X_train[:, feature_indices]
        X_val_subset = self.X_val[:, feature_indices]
        X_test_subset = self.X_test[:, feature_indices]
        
        # 创建Logistic Regression分类器
        lr_model = LogisticRegression(
            random_state=self.random_seed,
            max_iter=1000,
            solver='liblinear' if self.num_classes == 2 else 'lbfgs',
            multi_class='ovr' if self.num_classes > 2 else 'auto',
            class_weight='balanced'  # 自动平衡类别权重，类似XGBoost的scale_pos_weight
        )
        
        # 训练模型
        lr_model.fit(X_train_subset, self.y_train)
        
        # 预测
        val_predictions = lr_model.predict(X_val_subset)
        test_predictions = lr_model.predict(X_test_subset)
        
        val_acc = accuracy_score(self.y_val, val_predictions)
        test_acc = accuracy_score(self.y_test, test_predictions)
        
        # 计算AUC (仅二分类)
        if self.num_classes == 2:
            val_probs = lr_model.predict_proba(X_val_subset)[:, 1]
            test_probs = lr_model.predict_proba(X_test_subset)[:, 1]
            val_auc = roc_auc_score(self.y_val, val_probs)
            test_auc = roc_auc_score(self.y_test, test_probs)
        else:
            val_auc = 0.0
            test_auc = 0.0
        
        # 计算F1分数
        val_f1 = f1_score(self.y_val, val_predictions, average='weighted')
        test_f1 = f1_score(self.y_test, test_predictions, average='weighted')
        
        return {
            'val_acc': val_acc,
            'val_auc': val_auc,
            'val_f1': val_f1,
            'test_acc': test_acc,
            'test_auc': test_auc,
            'test_f1': test_f1
        }
    
    def run_comparison_experiment(self, max_features=None, step=1, num_repeats=1):
        """运行模型对比实验"""
        if max_features is None:
            max_features = self.total_features
        
        max_features = min(max_features, self.total_features)
        feature_range = list(range(1, max_features + 1, step))
        
        print(f"\n🚀 开始模型对比实验...")
        print(f"🔬 对比模型: MLP, XGBoost, Logistic Regression")
        print(f"📏 特征范围: 1 到 {max_features} (步长: {step})")
        print(f"🔄 重复次数: {num_repeats}")
        print(f"📊 总实验数: {len(feature_range) * num_repeats * 3}")
        print("-" * 80)
        
        all_results = []
        
        # 主进度条
        feature_progress = tqdm(feature_range, desc="特征数量", 
                               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}')
        
        for num_features in feature_progress:
            feature_progress.set_description(f"测试 {num_features:2d} 个特征")
            
            # 存储这个特征数量下的所有重复结果
            mlp_results = []
            xgb_results = []
            lr_results = []
            
            for repeat in range(num_repeats):
                # 选择特征
                feature_indices = self.select_top_features(num_features)
                
                # 训练三个模型
                mlp_metrics = self.train_mlp(feature_indices)
                xgb_metrics = self.train_xgboost(feature_indices)
                lr_metrics = self.train_logistic_regression(feature_indices)
                
                mlp_results.append(mlp_metrics)
                xgb_results.append(xgb_metrics)
                lr_results.append(lr_metrics)
            
            # 计算平均结果
            result = {'num_features': num_features}
            for model_name, results in [('mlp', mlp_results), ('xgb', xgb_results), ('lr', lr_results)]:
                for metric in ['val_acc', 'test_acc', 'val_auc', 'test_auc', 'val_f1', 'test_f1']:
                    values = [r[metric] for r in results]
                    result[f'{model_name}_{metric}_mean'] = np.mean(values)
                    result[f'{model_name}_{metric}_std'] = np.std(values)
            
            all_results.append(result)
            
            # 更新进度条显示
            best_acc = max(result['mlp_test_acc_mean'], 
                          result['xgb_test_acc_mean'], 
                          result['lr_test_acc_mean'])
            best_model_accs = [result['mlp_test_acc_mean'], 
                              result['xgb_test_acc_mean'], 
                              result['lr_test_acc_mean']]
            best_model = ['MLP', 'XGB', 'LR'][best_model_accs.index(best_acc)]
            
            feature_progress.set_postfix({
                'Best': f'{best_model}({best_acc:.3f})',
                'MLP': f'{result["mlp_test_acc_mean"]:.3f}',
                'XGB': f'{result["xgb_test_acc_mean"]:.3f}',
                'LR': f'{result["lr_test_acc_mean"]:.3f}'
            })
        
        return all_results
    
    def plot_comparison_results(self, results, save_path=None):
        """绘制模型对比结果"""
        if save_path:
            base_name = save_path.replace('.png', '')
            comparison_path = f"{base_name}_comparison.png"
            individual_path = f"{base_name}_individual.png"
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            comparison_path = f"model_comparison_{timestamp}.png"
            individual_path = f"model_individual_{timestamp}.png"
        
        self.plot_models_comparison(results, comparison_path)
        self.plot_individual_performance(results, individual_path)
    
    def plot_models_comparison(self, results, save_path=None):
        colors = {
            'mlp': '#3498DB',
            'xgb': '#E74C3C',
            'lr': '#2ECC71'
        }
        
        model_names = {
            'mlp': 'MLP',
            'xgb': 'XGBoost', 
            'lr': 'Logistic Regression'
        }
        
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('Model Performance Comparison: MLP vs XGBoost vs Logistic Regression', 
                     fontsize=20, fontweight='bold', y=0.95)
        
        x = [r['num_features'] for r in results]
        ax1 = plt.subplot(2, 2, 1)
        for model in ['mlp', 'xgb', 'lr']:
            test_acc_mean = [r[f'{model}_test_acc_mean'] for r in results]
            test_acc_std = [r[f'{model}_test_acc_std'] for r in results]
            ax1.errorbar(x, test_acc_mean, yerr=test_acc_std, 
                        marker='o', markersize=8, linewidth=3, capsize=6,
                        label=model_names[model], color=colors[model], alpha=0.9)
        
        ax1.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Test Accuracy vs Number of Features', fontsize=14, fontweight='bold', pad=15)
        ax1.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        ax2 = plt.subplot(2, 2, 2)
        if self.num_classes == 2:
            for model in ['mlp', 'xgb', 'lr']:
                test_auc_mean = [r[f'{model}_test_auc_mean'] for r in results]
                test_auc_std = [r[f'{model}_test_auc_std'] for r in results]
                ax2.errorbar(x, test_auc_mean, yerr=test_auc_std,
                            marker='s', markersize=8, linewidth=3, capsize=6,
                            label=model_names[model], color=colors[model], alpha=0.9)
            ax2.set_ylabel('Test AUC', fontsize=12, fontweight='bold')
            ax2.set_title('Test AUC vs Number of Features', fontsize=14, fontweight='bold', pad=15)
        else:
            for model in ['mlp', 'xgb', 'lr']:
                test_f1_mean = [r[f'{model}_test_f1_mean'] for r in results]
                test_f1_std = [r[f'{model}_test_f1_std'] for r in results]
                ax2.errorbar(x, test_f1_mean, yerr=test_f1_std,
                            marker='s', markersize=8, linewidth=3, capsize=6,
                            label=model_names[model], color=colors[model], alpha=0.9)
            ax2.set_ylabel('Test F1 Score', fontsize=12, fontweight='bold')
            ax2.set_title('F1 Score vs Number of Features', fontsize=14, fontweight='bold', pad=15)
        
        ax2.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        ax3 = plt.subplot(2, 2, 3)
        for model in ['mlp', 'xgb', 'lr']:
            val_acc = [r[f'{model}_val_acc_mean'] for r in results]
            test_acc = [r[f'{model}_test_acc_mean'] for r in results]
            ax3.plot(x, val_acc, marker='o', linewidth=2, markersize=6, 
                    color=colors[model], label=f'{model_names[model]} (Val)', alpha=0.7, linestyle='--')
            ax3.plot(x, test_acc, marker='s', linewidth=3, markersize=6,
                    color=colors[model], label=f'{model_names[model]} (Test)', alpha=0.9)
        
        ax3.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax3.set_title('Validation vs Test Accuracy', fontsize=14, fontweight='bold', pad=15)
        ax3.legend(fontsize=9, frameon=True, fancybox=True, shadow=True, ncol=2)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        ax4 = plt.subplot(2, 2, 4)
        win_counts = {'MLP': 0, 'XGBoost': 0, 'LogReg': 0}
        for r in results:
            test_accs = {
                'MLP': r['mlp_test_acc_mean'],
                'XGBoost': r['xgb_test_acc_mean'],
                'LogReg': r['lr_test_acc_mean']
            }
            best_model = max(test_accs, key=test_accs.get)
            win_counts[best_model] += 1
        
        model_names_short = list(win_counts.keys())
        counts = list(win_counts.values())
        bars = ax4.bar(model_names_short, counts, 
                      color=[colors['mlp'], colors['xgb'], colors['lr']], 
                      alpha=0.8, edgecolor='white', linewidth=2)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax4.set_ylabel('Number of Wins', fontsize=12, fontweight='bold')
        ax4.set_title('Best Model Count', fontsize=14, fontweight='bold', pad=15)
        ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"📊 模型对比图已保存到: {save_path}")
        
        plt.show()
    
    def plot_individual_performance(self, results, save_path=None):
        individual_colors = {
            'mlp': '#9B59B6',
            'xgb': '#F39C12',
            'lr': '#1ABC9C'
        }
        
        model_names = {
            'mlp': 'MLP',
            'xgb': 'XGBoost', 
            'lr': 'Logistic Regression'
        }
        
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('Individual Model Performance Analysis', 
                     fontsize=20, fontweight='bold', y=0.95)
        
        x = [r['num_features'] for r in results]
        model_keys = ['mlp', 'xgb', 'lr']
        
        for i, model in enumerate(model_keys):
            ax1 = plt.subplot(3, 2, i*2 + 1)
            val_acc = [r[f'{model}_val_acc_mean'] for r in results]
            test_acc = [r[f'{model}_test_acc_mean'] for r in results]
            val_std = [r[f'{model}_val_acc_std'] for r in results]
            test_std = [r[f'{model}_test_acc_std'] for r in results]
            
            ax1.errorbar(x, val_acc, yerr=val_std, marker='o', linewidth=3, markersize=8, 
                        color=individual_colors[model], label='Validation', alpha=0.7, 
                        linestyle='--', capsize=4)
            ax1.errorbar(x, test_acc, yerr=test_std, marker='s', linewidth=3, markersize=8,
                        color=individual_colors[model], label='Test', alpha=0.9, capsize=4)
            
            ax1.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
            ax1.set_title(f'{model_names[model]}', fontsize=14, fontweight='bold', pad=15)
            ax1.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            
            ax2 = plt.subplot(3, 2, i*2 + 2)
            
            test_acc_data = [r[f'{model}_test_acc_mean'] for r in results]
            test_f1_data = [r[f'{model}_test_f1_mean'] for r in results]
            
            ax2.plot(x, test_acc_data, marker='o', linewidth=3, markersize=8,
                    color=individual_colors[model], label='Accuracy', alpha=0.9)
            ax2.plot(x, test_f1_data, marker='s', linewidth=3, markersize=8,
                    color=individual_colors[model], label='F1 Score', alpha=0.7, linestyle='--')
            
            if self.num_classes == 2:
                test_auc_data = [r[f'{model}_test_auc_mean'] for r in results]
                ax2.plot(x, test_auc_data, marker='^', linewidth=3, markersize=8,
                        color=individual_colors[model], label='AUC', alpha=0.8, linestyle=':')
            
            ax2.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax2.set_title(f'{model_names[model]} - Multiple Metrics', fontsize=14, fontweight='bold', pad=15)
            ax2.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"📊 单独性能图已保存到: {save_path}")
        
        plt.show()
    
    def print_summary(self, results):
        """打印实验总结"""
        print("\n" + "="*80)
        print("🏆 模型性能对比总结")
        print("="*80)
        best_performances = {}
        for model in ['mlp', 'xgb', 'lr']:
            model_results = []
            for r in results:
                model_results.append({
                    'num_features': r['num_features'],
                    'test_acc': r[f'{model}_test_acc_mean'],
                    'test_auc': r[f'{model}_test_auc_mean'],
                    'test_f1': r[f'{model}_test_f1_mean']
                })
            
            best_result = max(model_results, key=lambda x: x['test_acc'])
            best_performances[model] = best_result
        
        model_names = {'mlp': 'MLP', 'xgb': 'XGBoost', 'lr': 'Logistic Regression'}
        
        for model, name in model_names.items():
            best = best_performances[model]
            print(f"\n📈 {name}:")
            print(f"   最佳准确率: {best['test_acc']:.4f} (特征数: {best['num_features']})")
            if self.num_classes == 2:
                print(f"   最佳AUC: {best['test_auc']:.4f}")
            print(f"   最佳F1分数: {best['test_f1']:.4f}")
        
        overall_best = max(best_performances.items(), key=lambda x: x[1]['test_acc'])
        print(f"\n🏆 整体最佳模型: {model_names[overall_best[0]]}")
        print(f"   准确率: {overall_best[1]['test_acc']:.4f}")
        print(f"   特征数: {overall_best[1]['num_features']}")
        
        print(f"\n📊 胜负统计:")
        win_counts = {'mlp': 0, 'xgb': 0, 'lr': 0}
        for r in results:
            test_accs = {
                'mlp': r['mlp_test_acc_mean'],
                'xgb': r['xgb_test_acc_mean'],
                'lr': r['lr_test_acc_mean']
            }
            best_model = max(test_accs, key=test_accs.get)
            win_counts[best_model] += 1
        
        total_tests = len(results)
        for model, name in model_names.items():
            percentage = (win_counts[model] / total_tests) * 100
            print(f"   {name}: {win_counts[model]}/{total_tests} ({percentage:.1f}%)")
    
    def save_results(self, results, filename_prefix="model_comparison"):
        """保存实验结果"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        csv_path = f"{filename_prefix}_{timestamp}.csv"
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        
        json_path = f"{filename_prefix}_{timestamp}.json"
        save_data = {
            'experiment_info': {
                'data_path': self.data_path,
                'total_features': self.total_features,
                'num_classes': self.num_classes,
                'device': str(self.device),
                'random_seed': self.random_seed
            },
            'results': results
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 结果已保存:")
        print(f"   CSV: {csv_path}")
        print(f"   JSON: {json_path}")
        
        return csv_path, json_path

def main():
    data_path = "/root/autodl-tmp/MISS_lzy/dataset/HI/mcar_HI_0.1.csv"
    
    max_features = None
    step = 1
    num_repeats = 1
    
    print("🔬 模型性能对比: MLP vs XGBoost vs Logistic Regression")
    print("="*70)
    
    comparator = ModelComparator(data_path)
    comparator.load_and_preprocess_data()
    
    start_time = time.time()
    results = comparator.run_comparison_experiment(
        max_features=max_features,
        step=step,
        num_repeats=num_repeats
    )
    end_time = time.time()
    
    print(f"\n⏱️  实验完成，耗时: {end_time - start_time:.2f} 秒")
    
    csv_path, json_path = comparator.save_results(results)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    comparison_plot_path = f"model_comparison_{timestamp}.png"
    individual_plot_path = f"model_individual_{timestamp}.png"
    
    comparator.plot_models_comparison(results, comparison_plot_path)
    comparator.plot_individual_performance(results, individual_plot_path)
    comparator.print_summary(results)

if __name__ == "__main__":
    main() 