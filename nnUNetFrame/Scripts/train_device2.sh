#!/bin/bash

#预处理：nnUNetv2_plan_and_preprocess -d 002 --verify_dataset_integrity

# ================== 在这里一键修改 ==================
DATASET_ID=002              # 修改数据集 ID，例如 10、17、24 等
UNET_CONFIGURATION="2d"
# 2d：2D U-Net（用于 2D 和 3D 数据集）
# 3d_fullres：在高图像分辨率上运行的 3D U-Net（仅适用于 3D 数据集）
# 3d_lowres → 3d_cascade_fullres：3D U-Net 级联
FLAG_DIR="/data/taoyusong/nnUNet_IVUS/nnUNet/nnUNetFrame/DATASET/nnUNet_results/Flag_${DATASET_ID}"
MAX_C_RETRIES=100  # **最大重试次数**
# ===================================================

export CUDA_VISIBLE_DEVICES=2

mkdir -p "$FLAG_DIR"

for f in 2; do
    if [ -f "${FLAG_DIR}/fold_${f}_completed" ]; then
        echo -e "\n\033[1;42m\033[97m  ===>>> FOLD $f 已经训练完成，跳过 ===>>>  \033[0m\n"
        continue
    fi

    c_retry_count=0
    use_c_mode=false

    echo -e "\n\033[1;44m\033[97m  ===>>>  开始训练 Fold $f  ===>>>  \033[0m\n"

    while true; do
        LOG_FILE="${FLAG_DIR}/temp_log_fold_${f}.txt"

        if [ "$use_c_mode" = true ]; then
            echo -e "\033[1;47m\033[34m  >>> 使用 --c 模式继续（第 $((c_retry_count))/$MAX_C_RETRIES 次）<<<\033[0m"
            nnUNetv2_train "$DATASET_ID" "$UNET_CONFIGURATION" "$f" --c 2>&1 | tee "$LOG_FILE"
        else
            echo -e "\033[1;47m\033[34m  >>> 重启训练 <<<\033[0m"
            nnUNetv2_train "$DATASET_ID" "$UNET_CONFIGURATION" "$f"  2>&1 | tee "$LOG_FILE"
        fi

        exit_code=${PIPESTATUS[0]}

        if [ $exit_code -eq 0 ]; then
            echo -e "\n\033[1;42m\033[97m  FOLD $f 训练成功！！！\033[0m\n"
            touch "${FLAG_DIR}/fold_${f}_completed"
            rm -f "$LOG_FILE"
            break
        else
            echo -e "\n\033[1;41m\033[97m  FOLD $f 训练失败！\033[0m\n"

            if [ $c_retry_count -ge $MAX_C_RETRIES ]; then
                echo -e "\033[1;41m\033[97m  已达最大 --c 重试次数，彻底放弃 Fold $f  \033[0m\n"
                rm -f "$LOG_FILE"
                break
            fi

            if grep -q "NaN detected, current_epoch=" "$LOG_FILE"; then
                current_epoch=$(grep "NaN detected, current_epoch=" "$LOG_FILE" | sed -E 's/.*current_epoch=([0-9]+).*/\1/' | tail -1)
                echo -e "\033[1;45m\033[97m  检测到 NaN！发生在 epoch $current_epoch  \033[0m"

                if [ "$current_epoch" -gt 10 ]; then
                    echo -e "\033[1;45m\033[97m  epoch > 10 \033[0m"
                    c_retry_count=$((c_retry_count + 1))
                    use_c_mode=true
                else
                    echo -e "\033[1;45m\033[97m  epoch ≤ 10 \033[0m"
                    use_c_mode=false
                fi
            else
                echo -e "\033[1;45m\033[97m  未知错误，重试  \033[0m"
            fi

            rm -f "$LOG_FILE"
            echo -e "\033[1;36m  3秒后重新开始训练...\033[0m\n"
            sleep 3
        fi
    done
done

echo -e "\n\033[1;45m\033[97m  目标 Fold 已处理完毕！训练脚本结束 \033[0m\n"