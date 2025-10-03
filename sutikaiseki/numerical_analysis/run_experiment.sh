#!/bin/bash

# 実行対象のPythonスクリプト
SCRIPT="experiment_gauss_newton_method.py"

# 入力画像と固定パラメータ
IMAGE="input/color/Lenna.bmp"
SCALE_TRUE=1.2
THETA_TRUE=15
THRESHOLD=1e-5

# scale_init のリスト
for S in 0.9 1.1 1.3 1.5; do
    # theta_init のリスト
    for T in 5 10 15 20 25; do
        echo "実行中: scale_init=$S, theta_init=$T"
        python3 "$SCRIPT" "$IMAGE" "$SCALE_TRUE" "$THETA_TRUE" --scale_init "$S" --theta_init "$T" --threshold "$THRESHOLD"
    done
done

echo "実験が完了しました"
