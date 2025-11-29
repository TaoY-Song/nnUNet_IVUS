#!/bin/bash

# ========================================
# 配置参数
# ========================================
DATASET_NAME="Dataset002_IVUS"
RESULTS_FOLDER="/data/taoyusong/nnUNet_IVUS/nnUNet/nnUNetFrame/DATASET/nnUNet_results/Dataset002_IVUS/nnUNetTrainer__nnUNetPlans__2d"
TRAINER_NAME="nnUNetTrainer"
PLANS_NAME="nnUNetPlans"
CONFIG="2d"
INPUT_FOLDER="/data/taoyusong/nnUNet_IVUS/nnUNet/nnUNetFrame/DATASET/nnUNet_inference/all_IVUS"
OUTPUT_FOLDER="/data/taoyusong/nnUNet_IVUS/nnUNet/nnUNetFrame/DATASET/nnUNet_inference/all_IVUS_output"
OUTPUT_FOLDER_PP="/data/taoyusong/nnUNet_IVUS/nnUNet/nnUNetFrame/DATASET/nnUNet_inference/all_IVUS_output_postprocessing"
PP_PKL_FILE="${RESULTS_FOLDER}/crossval_results_folds_0_1_2_3_4/postprocessing.pkl"
PLANS_JSON_FILE="${RESULTS_FOLDER}/crossval_results_folds_0_1_2_3_4/plans.json"

AVAILABLE_GPUS=(0 1 2 3)
GPU_COUNT=${#AVAILABLE_GPUS[@]}

# ========================================
# 信号处理 - 异常中断时保留数据
# ========================================
cleanup() {
    echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] 检测到中断信号，正在清理并合并已处理的数据..."
    for gpu_id in "${AVAILABLE_GPUS[@]}"; do
        temp_output="$OUTPUT_FOLDER.part_$gpu_id"
        if [ -d "$temp_output" ] && [ -n "$(ls -A "$temp_output" 2>/dev/null)" ]; then
            echo "合并 GPU $gpu_id 的临时文件..."
            mv "$temp_output"/* "$OUTPUT_FOLDER"/ 2>/dev/null || true
        fi
        rm -rf "$temp_output"
        rm -rf "$INPUT_FOLDER.part_$gpu_id"
    done
    echo "清理完成，已保留已处理的数据。"
    exit 1
}
trap cleanup SIGINT SIGTERM

# ========================================
# 初始化
# ========================================
mkdir -p "$OUTPUT_FOLDER" "$OUTPUT_FOLDER_PP"

# ========================================
# 筛选未处理的文件（断点续传）
# ========================================
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 检查已处理的文件..."
files_to_process=()
for file in "$INPUT_FOLDER"/*; do
    [ -e "$file" ] || continue
    filename=$(basename "$file")
    # 假设输出文件名与输入文件名相同（nnUNet默认行为）
    output_file="$OUTPUT_FOLDER/$filename"
    if [ ! -f "$output_file" ]; then
        files_to_process+=("$file")
    fi
done

total_files=${#files_to_process[@]}
echo "总计: $(ls "$INPUT_FOLDER"/* 2>/devnull | wc -l) 个文件，已处理: $(ls "$OUTPUT_FOLDER"/* 2>/dev/null | wc -l) 个文件，待处理: $total_files 个文件"

# ========================================
# 无文件需处理，直接后处理
# ========================================
if [ $total_files -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 所有文件已处理，跳过预测步骤，直接执行后处理..."
    if [ -f "$PP_PKL_FILE" ] && [ -f "$PLANS_JSON_FILE" ]; then
        nnUNetv2_apply_postprocessing -i "$OUTPUT_FOLDER" -o "$OUTPUT_FOLDER_PP" \
            -pp_pkl_file "$PP_PKL_FILE" -np 8 -plans_json "$PLANS_JSON_FILE"
        echo "后处理完成！"
    else
        echo "警告: 后处理文件不存在，请检查路径。"
    fi
    exit 0
fi

# ========================================
# 并行预测
# ========================================
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始并行预测..."
files_per_gpu=$(( (total_files + GPU_COUNT - 1) / GPU_COUNT ))

for gpu_idx in "${!AVAILABLE_GPUS[@]}"; do
    gpu_id=${AVAILABLE_GPUS[$gpu_idx]}
    start_idx=$((gpu_idx * files_per_gpu))
    end_idx=$((start_idx + files_per_gpu))
    
    temp_input="$INPUT_FOLDER.part_$gpu_id"
    temp_output="$OUTPUT_FOLDER.part_$gpu_id"
    mkdir -p "$temp_input" "$temp_output"
    
    # 为当前GPU分配文件（创建软链接）
    for ((i=start_idx; i<end_idx && i<total_files; i++)); do
        ln -sf "$(realpath "${files_to_process[$i]}")" "$temp_input/"
    done
    
    # 检查是否有文件分配给当前GPU
    if [ -z "$(ls -A "$temp_input")" ]; then
        rm -rf "$temp_input" "$temp_output"
        continue
    fi
    
    echo "  GPU $gpu_id 处理 ${#temp_input[@]} 个文件..."
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    nnUNetv2_predict -d "$DATASET_NAME" -i "$temp_input" -o "$temp_output" \
        -f 0 1 2 3 4 -tr "$TRAINER_NAME" -c "$CONFIG" -p "$PLANS_NAME" &
done

wait

# ========================================
# 合并清理
# ========================================
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 合并结果并清理临时文件..."
for gpu_id in "${AVAILABLE_GPUS[@]}"; do
    temp_output="$OUTPUT_FOLDER.part_$gpu_id"
    if [ -d "$temp_output" ] && [ -n "$(ls -A "$temp_output" 2>/dev/null)" ]; then
        mv "$temp_output"/* "$OUTPUT_FOLDER"/ 2>/dev/null || true
    fi
    rm -rf "$temp_output"
    rm -rf "$INPUT_FOLDER.part_$gpu_id"
done

# ========================================
# 后处理
# ========================================
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 执行后处理..."
if [ -f "$PP_PKL_FILE" ] && [ -f "$PLANS_JSON_FILE" ]; then
    nnUNetv2_apply_postprocessing -i "$OUTPUT_FOLDER" -o "$OUTPUT_FOLDER_PP" \
        -pp_pkl_file "$PP_PKL_FILE" -np 8 -plans_json "$PLANS_JSON_FILE"
    echo "后处理完成！"
else
    echo "警告: 后处理文件不存在，请检查路径。"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 全部任务完成！"