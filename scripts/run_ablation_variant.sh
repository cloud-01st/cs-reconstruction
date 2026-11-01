#!/bin/bash
# 消融实验单变体：与主模型（ours）完全相同的训练配方
# 用法: bash scripts/run_ablation_variant.sh <FLAG> <VARIANT>
#   FLAG:    --no-learnable-sampling / --no-multiscale / --no-attention
#   VARIANT: noLS / noMS / noAttn
set -e
cd "D:/毕业设计/code"
FLAG=$1
VARIANT=$2

echo "=============================================="
echo "[消融] $VARIANT  开始（100 epoch 主训练）"
echo "=============================================="
.venv/Scripts/python.exe -m src.train --model ours --rate 0.25 --epochs 100 $FLAG 2>&1 | tail -6

echo "=============================================="
echo "[消融] $VARIANT  进入精调（10 epoch, lr=2e-4）"
echo "=============================================="
.venv/Scripts/python.exe -m src.train --model ours --rate 0.25 --epochs 10 --lr 2e-4 \
    --init-from "checkpoints/ours_${VARIANT}_r25.pth" $FLAG 2>&1 | tail -6

echo "=============================================="
echo "[消融] $VARIANT  训练完成，评估中"
echo "=============================================="
.venv/Scripts/python.exe scripts/eval_model.py --model "ours_${VARIANT}" --rates 0.25 2>&1 | tail -12

echo "[消融] $VARIANT  全流程完成 ✓"
