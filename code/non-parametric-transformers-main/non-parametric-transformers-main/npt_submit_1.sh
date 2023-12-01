#!/bin/bash

# 定义要循环的参数值
# 循环嵌套，遍历参数组合
for ds in "Gesture"; do
  for mr in  "0.5"; do
    for mt in "mcar_"; do
      for np in "0" "149669" "52983"; do
          for im in "mean"; do
            # 构建命令行参数
            cmd="python run_1.py --data_set $ds --missingrate $mr --missingtype $mt --np_seed $np --torch_seed $np --imp $im --exp_device "cuda:1""

            # 执行命令
            echo "Running: $cmd"
            $cmd
          done
        done
      done
    done
  done
done
