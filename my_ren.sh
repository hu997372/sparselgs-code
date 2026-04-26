#! /bin/bash

# DATASET=teatime
# DATASET=waldo_kitchen
# DATASET=ramen
# DATASET=figurines
# DATASET=bed
# DATASET=bench
# DATASET=sofa
DATASET=room
# DATASET=office_desk
# DATASET=lawn
lerf_ovs="teatime waldo_kitchen ramen figurines"

if echo "$lerf_ovs" | grep -q "\b$DATASET\b"; then
    N_VIEWS=4
else
    N_VIEWS=3
fi
feature_level=2

python my_render.py -m ./output/$DATASET/${N_VIEWS}_views_${feature_level} --dataname $DATASET --include_feature
# python my_render.py -m ./output/bench/3_views_2 --include_feature