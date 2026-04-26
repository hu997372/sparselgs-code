#! /bin/bash

# DATASET=teatime
DATASET=waldo_kitchen
# DATASET=ramen
# DATASET=figurines
# DATASET=bed
# DATASET=bench
# DATASET=sofa
# DATASET=room
# DATASET=office_desk
# DATASET=lawn
lerf_ovs="teatime waldo_kitchen ramen figurines"

if echo "$lerf_ovs" | grep -q "\b$DATASET\b"; then
    N_VIEWS=4
else
    N_VIEWS=3
fi


IMAGE_PATH=./data/${DATASET}/
SOURCE_PATH=./data/${DATASET}/dust3r_${N_VIEWS}_views/
MODEL_PATH=./output/${DATASET}/${N_VIEWS}_views
ckpt_path=${SOURCE_PATH}dust3r_${N_VIEWS}_views/chkpnt1000.pth
ITER=1000
level=3

# python ./coarse_init_eval.py --n_views $N_VIEWS --img_base_path $IMAGE_PATH --focal_avg

# python ./train_joint.py -s $SOURCE_PATH -m $MODEL_PATH --n_views $N_VIEWS --iter $ITER --feature_level ${level} --include_feature_get 0 --optim_pose
# # python ./train_joint.py -s $SOURCE_PATH -m $MODEL_PATH --n_views $N_VIEWS --iter $ITER --feature_level ${level} --include_feature_get 0
# # python ./train_joint.py -s ./data/lerf_teatime/dust3r_4_views/ -m ./output/lerf_teatime/4_views --n_views 4 --iter 1000 --feature_level 3 --include_feature_get 0

# if [ ! -d "${SOURCE_PATH}dust3r_${N_VIEWS}_views" ]; then
#     mkdir ${SOURCE_PATH}dust3r_${N_VIEWS}_views
# fi
# cp -r ${MODEL_PATH}_${level}/* ${SOURCE_PATH}dust3r_${N_VIEWS}_views
# # mv ${SOURCE_PATH}${N_VIEWS}_views_${level} ${SOURCE_PATH}dust3r_${N_VIEWS}_views

# python preprocess.py --dataset_path $SOURCE_PATH
# # python preprocess.py --dataset_path ./data/waldo_kitchen/dust3r_4_views/

level_list=(2)
# level_list=(1 2 3)

for level in ${level_list[*]}
do
    python ./train_joint.py -s $SOURCE_PATH -m $MODEL_PATH --n_views $N_VIEWS --iter $ITER --start_checkpoint $ckpt_path --feature_level ${level} --include_feature_get 1
    # python ./train_joint.py -s ./data/lerf_teatime/dust3r_4_views/ -m ./output/lerf_teatime/4_views --n_views 4 --iter 1000 --start_checkpoint ./data/lerf_teatime/dust3r_4_views/dust3r_4_views/chkpnt1000.pth --feature_level 3 --include_feature_get 1
done

# python ./init_test_pose.py --img_base_path $IMAGE_PATH --n_views $N_VIEWS --focal_avg
# python ./init_test_pose.py --img_base_path ./data/lerf_teatime/ --n_views 4 --focal_avg

for level in ${level_list[*]}
do
    # python render.py -s $SOURCE_PATH -m ${MODEL_PATH}_${level} --n_views $N_VIEWS --optim_test_pose_iter 500
    # python render.py -s ./data/lerf_teatime/dust3r_4_views/ -m ./output/lerf_teatime/4_views_3 --n_views 4 --optim_test_pose_iter 500

    python render.py -s $SOURCE_PATH -m ${MODEL_PATH}_${level} --n_views $N_VIEWS --optim_test_pose_iter 500 --include_feature
done