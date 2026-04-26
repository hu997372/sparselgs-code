#!/bin/bash
CASE_NAME='teatime'
# CASE_NAME='ramen'
# CASE_NAME='waldo_kitchen'
# CASE_NAME='figurines'
lerf_ovs="teatime waldo_kitchen ramen figurines"

if echo "$lerf_ovs" | grep -q "\b$DATASET\b"; then
    N_VIEWS=4
else
    N_VIEWS=3
fi

SOURCE_PATH=./data/${CASE_NAME}/dust3r_${N_VIEWS}_views/
root_path='..'
gt_folder='../download/lerf_ovs/label'

python evaluate_iou_loc.py --dataset_name ${CASE_NAME} --feat_dir ${root_path}/output/${CASE_NAME} --output_dir ${root_path}/eval_result --mask_thresh 0.6 --json_folder ${gt_folder}
# python evaluate_iou_loc_ab.py --dataset_name ${CASE_NAME} --feat_dir ${root_path}/output/${CASE_NAME} --output_dir ${root_path}/eval_result --mask_thresh 0.5 --json_folder ${gt_folder}