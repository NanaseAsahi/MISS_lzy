import json
import pandas as pd
from pathlib import Path
import argparse
import sys

"""
用来生成info.json文件

use:
cd /home/lzy/data/MISS2/code/tabular/tabular/tabular-dl-num-embeddings-main-our
python -m  lib.make_info_json
"""

# cd /home/lzy/data/MISS2/code/tabular/tabular/tabular-dl-num-embeddings-main-our
# python -m  lib.make_info_json
# 生成目录下没有的info.json
# 暂时不使用
# os.environ['PROJECT_DIR'] = "/home/lsw/tabular/tabular-dl-num-embeddings-main-our"
base_data_path = Path('/home/lzy/data/MISS2/dataset')
default_output_path = Path('/home/lzy/data/MISS2/data')

datasets = ['gas', 'HI', 'News', 'temperature']
missingtypes = ['mcar', 'mar_p', 'mnar_p']
missingrates = [0.1, 0.3, 0.5, 0.7, 0.9]

def make_by_parser():
    # todo 还未完成 7.30
    parser = argparse.ArgumentParser(description="使用命令行参数生成info.json")

    parser.add_argument("--path", type=str, required=True, help="the path of dataset")
    parser.add_argument("--task_type_type", type=str, required=True, choices=['binclass', 'multiclass', 'regression'])
    parser.add_argument("--output_path", type=str, required=False, defalut=default_output_path, help="the path to output info.json")

    args = parser.parse_args()

    print(f'path: {args.path}')
    print(f'task_type: {args.task_type}')

    if args.output_path is None:
        args.output_path = args.path.replace('.csv', '_info.json')

    print(f'output_path: {args.output_path}')

def make_info_json(data_path:Path):
    # 主要使用这个函数
    for dataset in datasets:
        for missingtype in missingtypes:
            for missingrate in missingrates:
                data_path = base_data_path / dataset / f'{missingtype}_{dataset}_{missingrate}.csv'
                output_path = default_output_path / dataset / f'{missingtype}_{missingrate}' / 'info.json'

                data = pd.read_csv(data_path)

                # print(data.head())
                """
                正常判断
                """
                # if 'label' in data.columns:
                #     label = data['label']
                # elif 'target' in data.columns:
                #     label = data['target']
                # elif 'Label' in data.columns:
                #     label = data['Label']
                # else: label = None
 
                # if label is None:
                #     task_type = 'regression'
                # else:
                #     task_type = 'binclass' if label.nunique() == 2 else 'multiclass'
                
                # n_classes = label.nunique() if task_type == 'multiclass' else 2 if task_type == 'binclass' else None

                """
                本文中
                Higgs Covertype 用于分类
                Temp Gas 用于回归
                """
                if dataset == 'HI' or dataset == 'News':
                    if 'label' in data.columns:
                        label = data['label']
                    elif 'target' in data.columns:
                        label = data['target']
                    elif 'Labels' in data.columns:
                        label = data['Labels']
                    elif 'shares' in data.columns:
                        label = data['shares']
                    else: label = None

                    if label is not None:
                        task_type = 'binclass' if label.nunique() == 2 else 'multiclass'
                    else:
                        task_type = 'regression'

                    n_classes = label.nunique() if task_type == 'multiclass' else 2 if task_type == 'binclass' else None

                elif dataset == 'temperature' or dataset == 'gas':
                    task_type = 'regression'

                n_classes = label.nunique() if task_type == 'multiclass' else 2 if task_type == 'binclass' else None

                info = {
                    # int 64无法被序列化
                    'task_type': task_type,
                    'n_classes': int(n_classes) if n_classes is not None else None,
                    'n_features': int(data.shape[1] - 1),                 
                    'n_samples': int(data.shape[0]),                     
                    'n_missing': int(data.isnull().sum().sum()),          
                    'missing_rate': float(data.isnull().mean().mean()),   
                    'missing_type': missingtype,
                }

                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(info, f, indent=4)

                print(f'success: save info.json to {output_path}')




if __name__ == "__main__":
    make_info_json(base_data_path)