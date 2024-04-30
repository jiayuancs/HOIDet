#!/bin/bash

# 检查是否有predict目录
if [ ! -d "./predict" ]; then
    echo "Error: Directory './predict' not found."
    exit 1
fi

# 获取所有以pred_开头的目录，并按照数字大小排序
dirs=$(ls -d ./predict/pred_* 2>/dev/null | sort -V)

# 检查是否有匹配的目录
if [ -z "$dirs" ]; then
    echo "Error: No directories found matching pattern 'pred_*' in './predict'."
    exit 1
fi

# 获取最新目录
max_dir=$(echo "$dirs" | tail -n 1)
echo "The latest results are saved in $max_dir"

# 检查最新目录下是否存在.txt文件
txt_files=$(ls "$max_dir"/*.txt 2>/dev/null)

# 输出.txt文件内容
if [ -n "$txt_files" ]; then
    cat $max_dir/*.txt
    python echoeval.py --dir $max_dir
else
    echo "There is no .txt file in this directory"
fi
