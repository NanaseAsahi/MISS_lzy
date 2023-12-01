#!/bin/bash

# 定义要循环的参数值
# 循环嵌套，遍历参数组合
for ds in "HI" "News"; do
  for mr in  "0.5"; do
    for mt in "mcar_"; do
      for np in "0" "149669" "52983"; do
        for ts in "0" "149669" "52983"; do
          for im in "mean" "_miwae" "_notmiwae" "_gain" "_missforest"; do
            # 构建命令行参数
            cmd="python run.py --data_set $ds --missingrate $mr --missingtype $mt --np_seed $np --torch_seed $ts --imp $im"

            # 执行命令
            echo "Running: $cmd"
            $cmd
          done
        done
      done
    done
  done
done
