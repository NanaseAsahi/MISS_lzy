import pandas as pd
import numpy as np
from pathlib import Path
import random
import os
import json


base_data_path = Path('/home/lzy/data/MISS2/dataset')
default_output_path = Path('/home/lzy/data/MISS2/data')

datasets = ['gas', 'HI', 'News', 'temperature']
missingtypes = ['mcar', 'mar_p', 'mnar_p']
missingrates = [0.1, 0.3, 0.5, 0.7, 0.9]

# random.seed(0)
# np.random.seed(0)
seeds = [0, 149669, 52983]


def identify_cat_features(X: pd.DataFrame) -> list[bool]:
    # 将特征分为连续型和离散型

    # 获取每列的唯一值个数和类型 用于分辨是否为离散型特征
    nunique = X.nunique()
    types = X.dtypes

    cat_indicator = np.zeros(X.shape[1]).astype(bool)

    for col in X.columns:
        # 先通过类型直接判断 计算速度更快
        if  types[col] == 'object' or nunique[col] < 100:
            # cat_indicator[col] = True
            cat_indicator[X.columns.get_loc(col)] = True

    # 获得连续特征和离散特征的列名    
    cat_columns = X.columns[cat_indicator].tolist()
    con_columns = list(set(X.columns.tolist()) - set(cat_columns))

    cat_idxs = list(np.where(cat_indicator == True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))

    return cat_idxs, con_idxs, cat_columns, con_columns

def split_train_test_val(X, y, X_num, X_cat, seed=0):
    # csv文件中最后一行均为label/y
    random.seed(seed)
    np.random.seed(seed)

    X_copy = X.copy()

    X_copy['Set'] = np.random.choice(['train', 'valid', 'test'], p=[.8, .1, .1], size=X_copy.shape[0])
    train_index = X_copy[X_copy['Set'] == 'train'].index
    test_index = X_copy[X_copy['Set'] == 'test'].index
    val_index = X_copy[X_copy['Set'] == 'valid'].index

    result = {}

    if X_cat is not None:
        result['X_cat'] = {
            'train': X_cat[train_index],
            'test': X_cat[test_index],
            'val': X_cat[val_index]
        }

    if X_num is not None:
        result['X_num'] = {
            'train' : X_num[train_index],
            'test' : X_num[test_index],
            'val' : X_num[val_index],
        }

    result['y'] = {
        'train' : y[train_index],
        'test' : y[test_index],
        'val' : y[val_index],
    }
    

    return result, train_index, test_index, val_index


def save_npy(data_dict, output_dir):
    output_dir = Path(output_dir)
    
    # 确保目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存所有数据文件
    for data_type, data in data_dict.items():
        for split, array in data.items():
            file_name = f'{data_type}_{split}.npy'
            file_path = output_dir / file_name
            np.save(file_path, array)
            print(f'保存 {file_name} 到 {file_path}')
                

def process_single_dataset(csv_path, output_dir, seed=0):
    """
    处理单个CSV文件，生成对应的npy文件
    """
    # 读取CSV数据
    data = pd.read_csv(csv_path)
    
    # 分离特征和标签
    X = data.iloc[:, :-1]  # 所有列除了最后一列
    
    # 找到标签列
    possible_label_cols = ['label', 'target', 'Labels', 'shares']
    y_col = None
    for col in possible_label_cols:
        if col in data.columns:
            y_col = col
            break
    
    if y_col is None:
        # 如果没有找到标准的标签列名，使用最后一列
        y = data.iloc[:, -1].values
    else:
        y = data[y_col].values
    
    # 识别分类特征和连续特征
    cat_idxs, con_idxs, _, _ = identify_cat_features(X)
    
    # 准备数值特征和分类特征数组
    X_num = X.iloc[:, con_idxs].values if con_idxs else None
    X_cat = X.iloc[:, cat_idxs].values if cat_idxs else None
    
    # 按train4.py方式分割数据
    split_data, train_idx, valid_idx, test_idx = split_train_test_val(
        X, y, X_num, X_cat, seed=seed
    )
    
    # 保存npy文件
    save_npy(split_data, output_dir)
     
    # 生成并保存info.json
    task_type = determine_task_type(y)
    n_classes = len(np.unique(y)) if task_type != 'regression' else None
     
    info = {
        'task_type': task_type,
        'n_classes': n_classes,
        'seed': seed,
        'n_samples': {
            'train': len(train_idx),
            'val': len(valid_idx), 
            'test': len(test_idx)
        },
        'n_features': {
            'categorical': len(cat_idxs) if cat_idxs else 0,
            'numerical': len(con_idxs) if con_idxs else 0
        }
    }
     
    info_path = Path(output_dir) / 'info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
     
    print(f"训练集样本数: {len(train_idx)}")
    print(f"验证集样本数: {len(valid_idx)}")
    print(f"测试集样本数: {len(test_idx)}")
     
    return {
        'train_size': len(train_idx),
        'val_size': len(valid_idx),
        'test_size': len(test_idx),
        'n_num_features': len(con_idxs) if con_idxs else 0,
        'n_cat_features': len(cat_idxs) if cat_idxs else 0
    }



def determine_task_type(y):
    """确定任务类型"""
    unique_values = np.unique(y)
    if len(unique_values) == 2:
        return 'binclass'
    elif len(unique_values) > 2 and y.dtype in ['int64', 'int32', 'object']:
        return 'multiclass'
    else:
        return 'regression'

def main():
    total_files = len(datasets) * len(missingtypes) * len(missingrates)
    current_file = 0
    success_files = 0
    error_files = 0

    print(f"开始批量生成npy文件...")
    print(f"输入路径: {base_data_path}")
    print(f"输出路径: {default_output_path}")
    print(f"总计: {total_files} 个数据集")
    print(f"数据集: {datasets}")
    print(f"缺失类型: {missingtypes}")
    print(f"缺失率: {missingrates}")

    for dataset in datasets:
        for missingtype in missingtypes:
            for missingrate in missingrates:
                current_file += 1

                csv_filename = f'{missingtype}_{dataset}_{missingrate}.csv'
                csv_path = base_data_path / dataset / csv_filename
                output_dir = default_output_path / dataset / f'{missingtype}_{missingrate}'

                # 进度条
                print(f"\n[{current_file}/{total_files}] 🔄 {csv_filename}")

                if csv_path.exists():
                    try:
                        # 为每个seed生成对应的数据划分
                        for seed in seeds:
                            # 创建seed特定的输出目录
                            seed_output_dir = output_dir / f'seed_{seed}'
                            
                            # 将所有操作抽象成对单个数据集的操作
                            info = process_single_dataset(csv_path, seed_output_dir, seed=seed)
                            print(f'  seed={seed} 保存到 {seed_output_dir}')
                            print(f"    训练集: {info['train_size']}, 验证集: {info['val_size']}, 测试集: {info['test_size']}")
                        
                        print(f'成功处理！所有seeds数据已生成')
                        success_files += 1
                    
                    except Exception as e:
                        print(f'处理失败: {e}')
                        error_files += 1

                else:
                    print(f'⚠️  文件不存在: {csv_path}')
                    error_files += 1

    print(f'\n🎉 处理完成！')
    print(f'成功处理: {success_files} 个文件')
    print(f'处理失败: {error_files} 个文件')
    print(f'总计: {total_files} 个文件')






if __name__ == '__main__':
    main()