#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CASE_NAME="${CASE_NAME:-teatime}"
# CASE_NAME='ramen'
# CASE_NAME='waldo_kitchen'
# CASE_NAME='figurines'
lerf_ovs="teatime waldo_kitchen ramen figurines"

if echo "$lerf_ovs" | grep -q "\b$CASE_NAME\b"; then
    N_VIEWS=4
else
    N_VIEWS=3
fi

root_path='..'
gt_folder='../download/lerf_ovs/label'

"$PYTHON_BIN" evaluate_iou_loc.py \
    --dataset_name "${CASE_NAME}" \
    --feat_dir "${root_path}/output/${CASE_NAME}" \
    --output_dir "${root_path}/eval_result" \
    --mask_thresh 0.6 \
    --json_folder "${gt_folder}" \
    --n_views "${N_VIEWS}"
# "$PYTHON_BIN" evaluate_iou_loc_ab.py --dataset_name "${CASE_NAME}" --feat_dir "${root_path}/output/${CASE_NAME}" --output_dir "${root_path}/eval_result" --mask_thresh 0.5 --json_folder "${gt_folder}" --n_views "${N_VIEWS}"
