
import subprocess
import os
# 将参数列表存储为列表
missingrates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
bi_datasets = ['News', 'HI', 'Credit']
mul_datasets = ['Chess', 'Gesture', 'Letter']

# --data_name HI --missing_rate 0.4 --task binary
for data in bi_datasets:
    for missingrate in missingrates:

        # 使用命令行参数运行你的脚本文件，并获得输出
        command = 'python train.py --data_name {} --missing_rate {} --task binary'.format(data, missingrate)
        # os.system(command)
        output = subprocess.check_output(command, shell=True)

        # 打印输出
        print(output.decode('utf-8'))

for data in mul_datasets:
    for missingrate in missingrates:

        # 使用命令行参数运行你的脚本文件，并获得输出
        command = 'python train.py --data_name {} --missing_rate {} --task binary'.format(data, missingrate)
        output = subprocess.check_output(command, shell=True)

        # 打印输出
        print(output.decode('utf-8'))