# dataset_name=ramen
# dataset_name=waldo_kitchen
# dataset_name=figurines
dataset_name=teatime
# dataset_name=bed
# dataset_name=bench
# dataset_name=sofa
# dataset_name=room
# dataset_name=office_desk
# dataset_name=lawn
lerf_ovs="teatime waldo_kitchen ramen figurines"

if echo "$lerf_ovs" | grep -q "\b$dataset_name\b"; then
    N_VIEWS=4
else
    N_VIEWS=3
fi

dataset_path=../data/${dataset_name}/dust3r_${N_VIEWS}_views

cd autoencoder

rm -rf ${dataset_path}/language_features/*
rm -rf ${dataset_path}/language_features_dim3/*
mv ${dataset_path}/language_features_origin/* ${dataset_path}/language_features
mv ${dataset_path}/language_features_origin_dim3/* ${dataset_path}/language_features_dim3
rm -rf ${dataset_path}/language_features_origin
rm -rf ${dataset_path}/language_features_origin_dim3

python train.py --dataset_path $dataset_path --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name $dataset_name
# e.g. python train.py --dataset_path ../data/sofa --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name sofa

# get the 3-dims language feature of the scene
python test.py --dataset_path $dataset_path --dataset_name $dataset_name
# python test.py --dataset_path ../data/teatime/dust3r_4_views --dataset_name teatime

cd ..