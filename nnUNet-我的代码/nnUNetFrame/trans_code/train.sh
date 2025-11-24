#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

for f in 0 1 2 3 4; do
    if [ -f "fold_${f}_completed" ]; then
        echo "Fold $f already completed, skipping."
        continue
    fi

    echo "Running fold $f without --c..."

    while true; do
        # 运行训练并捕获输出和退出码
        output=$(nnUNetv2_train 10 3d_fullres "$f" 2>&1)
        exit_code=$?

        if [ $exit_code -eq 0 ]; then
            echo "Fold $f completed successfully."
            touch "fold_${f}_completed"
            break
        else
            echo "Fold $f failed, checking output..."

            # 检查是否包含 NaN 信息
            if echo "$output" | grep -q "NaN detected, current_epoch="; then
                # 提取 current_epoch= 后面的数字
                current_epoch=$(echo "$output" | grep "current_epoch=" | sed -E 's/.*current_epoch=([0-9]+).*/\1/' | head -1)

                echo "NaN detected at epoch $current_epoch."

                if [ "$current_epoch" -gt 10 ]; then
                    echo "Epoch enough, retrying with --c..."
                    nnUNetv2_train 10 3d_fullres "$f" --c
                    # 不管这一次的 --c 是否成功，都结束当前 fold（与原 bat 行为一致）
                    break
                else
                    echo "Epoch not enough, retrying without --c..."
                    # 继续循环，不加 --c 重试
                    continue
                fi
            else
                echo "Unknown error, retrying without --c..."
                # 继续循环，不加 --c 重试
                continue
            fi
        fi
    done
done