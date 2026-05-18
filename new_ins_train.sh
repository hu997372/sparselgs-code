#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Optional JSON config. Environment variables still win, e.g.
# CONFIG=configs/pipeline.example.json DATASET=teatime bash new_ins_train.sh
CONFIG="${CONFIG:-}"
if [[ -n "$CONFIG" ]]; then
    [[ -f "$CONFIG" ]] || { printf 'ERROR: Missing config file: %s\n' "$CONFIG" >&2; exit 1; }
    eval "$(
        "$PYTHON_BIN" - "$CONFIG" <<'PY'
import json
import shlex
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

values = {}

def put(name, value):
    if value is None:
        return
    if isinstance(value, bool):
        value = "1" if value else "0"
    elif isinstance(value, list):
        value = " ".join(str(v) for v in value)
    else:
        value = str(value)
    values[name] = value

top_level = {
    "DATASET": "DATASET",
    "dataset": "DATASET",
    "N_VIEWS": "N_VIEWS",
    "n_views": "N_VIEWS",
    "ITER": "ITER",
    "iterations": "ITER",
    "BASE_FEATURE_LEVEL": "BASE_FEATURE_LEVEL",
    "base_feature_level": "BASE_FEATURE_LEVEL",
    "FEATURE_LEVELS": "FEATURE_LEVELS",
    "feature_levels": "FEATURE_LEVELS",
}
for key, name in top_level.items():
    if key in data:
        put(name, data[key])

stage_names = {
    "dust3r_init": "RUN_DUST3R_INIT",
    "rgb_train": "RUN_RGB_TRAIN",
    "sam_clip": "RUN_SAM_CLIP",
    "autoencoder": "RUN_AUTOENCODER",
    "lang_fusion": "RUN_LANG_FUSION",
    "feature_train": "RUN_FEATURE_TRAIN",
    "render": "RUN_RENDER",
    "test_pose_init": "RUN_TEST_POSE_INIT",
    "copy_geometry_to_source": "COPY_GEOMETRY_TO_SOURCE",
    "optim_pose": "OPTIM_POSE",
}
for key, name in stage_names.items():
    if key in data and not isinstance(data[key], dict):
        put(name, data[key])
    if isinstance(data.get("stages"), dict) and key in data["stages"]:
        put(name, data["stages"][key])

path_names = {
    "dust3r_ckpt": "DUST3R_CKPT",
    "sam_ckpt": "SAM_CKPT",
    "start_checkpoint": "START_CHECKPOINT",
}
for key, name in path_names.items():
    if key in data:
        put(name, data[key])
    if isinstance(data.get("paths"), dict) and key in data["paths"]:
        put(name, data["paths"][key])

render_names = {
    "optimize_test_pose": "OPTIMIZE_TEST_POSE",
    "optim_test_pose_iter": "OPTIM_TEST_POSE_ITER",
    "feature_checkpoint_iteration": "FEATURE_CKPT_ITER",
}
for key, name in render_names.items():
    if key in data:
        put(name, data[key])
    if isinstance(data.get("render"), dict) and key in data["render"]:
        put(name, data["render"][key])

if "ae_lr" in data:
    put("AE_LR", data["ae_lr"])
if isinstance(data.get("autoencoder"), dict) and "lr" in data["autoencoder"]:
    put("AE_LR", data["autoencoder"]["lr"])

lang_fusion_names = {
    "feature_level": "LANG_FUSION_LEVEL",
    "refresh_origin": "LANG_FUSION_REFRESH_ORIGIN",
    "visualize": "LANG_FUSION_VISUALIZE",
}
for key, name in lang_fusion_names.items():
    top_key = f"lang_fusion_{key}"
    if top_key in data:
        put(name, data[top_key])
    if isinstance(data.get("lang_fusion"), dict) and key in data["lang_fusion"]:
        put(name, data["lang_fusion"][key])

for name, value in values.items():
    print(f'if [[ -z "${{{name}:-}}" ]]; then {name}={shlex.quote(value)}; fi')
PY
    )"
fi

# Main knobs. Override them from the command line, e.g.
# DATASET=teatime FEATURE_LEVELS="2 3" RUN_AUTOENCODER=1 bash new_ins_train.sh
DATASET="${DATASET:-waldo_kitchen}"
ITER="${ITER:-1000}"
BASE_FEATURE_LEVEL="${BASE_FEATURE_LEVEL:-3}"
FEATURE_LEVELS="${FEATURE_LEVELS:-2}"

# Stages. Defaults keep the old workflow: train geometry, make SAM/CLIP
# features, train semantic Gaussians, and render. DUSt3R init and autoencoder
# compression are optional because many runs reuse precomputed outputs.
RUN_DUST3R_INIT="${RUN_DUST3R_INIT:-0}"
RUN_RGB_TRAIN="${RUN_RGB_TRAIN:-1}"
RUN_SAM_CLIP="${RUN_SAM_CLIP:-1}"
RUN_AUTOENCODER="${RUN_AUTOENCODER:-0}"
RUN_LANG_FUSION="${RUN_LANG_FUSION:-0}"
RUN_FEATURE_TRAIN="${RUN_FEATURE_TRAIN:-1}"
RUN_RENDER="${RUN_RENDER:-1}"
RUN_TEST_POSE_INIT="${RUN_TEST_POSE_INIT:-0}"
COPY_GEOMETRY_TO_SOURCE="${COPY_GEOMETRY_TO_SOURCE:-1}"
OPTIM_POSE="${OPTIM_POSE:-1}"

DUST3R_CKPT="${DUST3R_CKPT:-./ckpt/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"
SAM_CKPT="${SAM_CKPT:-./ckpt/sam_vit_h_4b8939.pth}"
AE_LR="${AE_LR:-0.0007}"
LANG_FUSION_LEVEL="${LANG_FUSION_LEVEL:-}"
LANG_FUSION_REFRESH_ORIGIN="${LANG_FUSION_REFRESH_ORIGIN:-0}"
LANG_FUSION_VISUALIZE="${LANG_FUSION_VISUALIZE:-0}"
OPTIMIZE_TEST_POSE="${OPTIMIZE_TEST_POSE:-0}"
OPTIM_TEST_POSE_ITER="${OPTIM_TEST_POSE_ITER:-0}"
FEATURE_CKPT_ITER="${FEATURE_CKPT_ITER:-1000}"
USER_START_CHECKPOINT="${START_CHECKPOINT:-}"

LERF_OVS_DATASETS=(teatime waldo_kitchen ramen figurines)

log() {
    printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

run() {
    log "+ $*"
    "$@"
}

die() {
    printf '\nERROR: %s\n' "$*" >&2
    exit 1
}

require_file() {
    local path="$1"
    local hint="${2:-}"
    [[ -f "$path" ]] || die "Missing file: ${path}${hint:+. ${hint}}"
}

require_dir() {
    local path="$1"
    local hint="${2:-}"
    [[ -d "$path" ]] || die "Missing directory: ${path}${hint:+. ${hint}}"
}

is_lerf_ovs_dataset() {
    local name="$1"
    local item
    for item in "${LERF_OVS_DATASETS[@]}"; do
        [[ "$item" == "$name" ]] && return 0
    done
    return 1
}

if [[ -z "${N_VIEWS:-}" ]]; then
    if is_lerf_ovs_dataset "$DATASET"; then
        N_VIEWS=4
    else
        N_VIEWS=3
    fi
fi

read -r -a FEATURE_LEVEL_ARRAY <<< "$FEATURE_LEVELS"
[[ "${#FEATURE_LEVEL_ARRAY[@]}" -gt 0 ]] || die "FEATURE_LEVELS is empty"
LANG_FUSION_LEVEL="${LANG_FUSION_LEVEL:-${FEATURE_LEVEL_ARRAY[0]}}"

IMAGE_PATH="./data/${DATASET}"
SOURCE_PATH="${IMAGE_PATH}/dust3r_${N_VIEWS}_views"
MODEL_ROOT="./output/${DATASET}/${N_VIEWS}_views"
BASE_MODEL_PATH="${MODEL_ROOT}_${BASE_FEATURE_LEVEL}"
SOURCE_GEOMETRY_DIR="${SOURCE_PATH}/dust3r_${N_VIEWS}_views"
BASE_CKPT="${BASE_MODEL_PATH}/chkpnt${ITER}.pth"
COPIED_CKPT="${SOURCE_GEOMETRY_DIR}/chkpnt${ITER}.pth"

log "Dataset=${DATASET}, N_VIEWS=${N_VIEWS}, ITER=${ITER}"
log "Source=${SOURCE_PATH}"
log "Model root=${MODEL_ROOT}"
log "Feature levels=${FEATURE_LEVELS}"
if [[ "$RUN_LANG_FUSION" == "1" ]]; then
    log "Language fusion level=${LANG_FUSION_LEVEL}"
fi

NEED_START_CHECKPOINT=0
NEED_DIM3_FEATURES=0
if [[ "$RUN_FEATURE_TRAIN" == "1" ]]; then
    NEED_START_CHECKPOINT=1
    NEED_DIM3_FEATURES=1
fi
if [[ "$RUN_LANG_FUSION" == "1" ]]; then
    NEED_DIM3_FEATURES=1
fi
if [[ "$RUN_RENDER" == "1" ]]; then
    NEED_DIM3_FEATURES=1
fi

if [[ "$RUN_DUST3R_INIT" == "1" ]]; then
    require_dir "${IMAGE_PATH}/images" "DUSt3R initialization samples sparse views from this folder."
    require_file "$DUST3R_CKPT" "Set DUST3R_CKPT=/path/to/checkpoint if it is elsewhere."
    run "$PYTHON_BIN" ./coarse_init_eval.py \
        --model_path "$DUST3R_CKPT" \
        --n_views "$N_VIEWS" \
        --img_base_path "$IMAGE_PATH" \
        --focal_avg
else
    require_dir "${SOURCE_PATH}/images" \
        "Run with RUN_DUST3R_INIT=1, or prepare the sparse-view image folder first."
    require_dir "${SOURCE_PATH}/sparse/0" \
        "Run with RUN_DUST3R_INIT=1, or prepare DUSt3R/COLMAP files first."
fi

if [[ "$RUN_RGB_TRAIN" == "1" ]]; then
    rgb_train_cmd=(
        "$PYTHON_BIN" ./train_joint.py
        -s "$SOURCE_PATH"
        -m "$MODEL_ROOT"
        --n_views "$N_VIEWS"
        --iter "$ITER"
        --feature_level "$BASE_FEATURE_LEVEL"
        --include_feature_get 0
    )
    if [[ "$OPTIM_POSE" == "1" ]]; then
        rgb_train_cmd+=(--optim_pose)
    fi
    run "${rgb_train_cmd[@]}"
elif [[ "$NEED_START_CHECKPOINT" == "1" ]]; then
    if [[ -z "$USER_START_CHECKPOINT" ]]; then
        require_file "$BASE_CKPT" "RUN_RGB_TRAIN=0 expects the geometry checkpoint to already exist."
    fi
fi

START_CHECKPOINT=""
if [[ "$NEED_START_CHECKPOINT" == "1" ]]; then
    if [[ -n "$USER_START_CHECKPOINT" ]]; then
        START_CHECKPOINT="$USER_START_CHECKPOINT"
    elif [[ "$COPY_GEOMETRY_TO_SOURCE" == "1" ]]; then
        require_file "$BASE_CKPT" "Geometry training should produce this checkpoint."
        mkdir -p "$SOURCE_GEOMETRY_DIR"
        run cp -a "${BASE_MODEL_PATH}/." "$SOURCE_GEOMETRY_DIR/"
        START_CHECKPOINT="$COPIED_CKPT"
    else
        require_file "$BASE_CKPT" "Geometry training should produce this checkpoint."
        START_CHECKPOINT="$BASE_CKPT"
    fi
    require_file "$START_CHECKPOINT" "Feature training uses this RGB geometry checkpoint."
fi

if [[ "$RUN_SAM_CLIP" == "1" ]]; then
    require_file "$SAM_CKPT" "Set SAM_CKPT=/path/to/sam_vit_h_4b8939.pth if it is elsewhere."
    run "$PYTHON_BIN" ./preprocess.py \
        --dataset_path "$SOURCE_PATH" \
        --sam_ckpt_path "$SAM_CKPT"
elif [[ "$RUN_AUTOENCODER" == "1" ]]; then
    require_dir "${SOURCE_PATH}/language_features" \
        "RUN_AUTOENCODER=1 needs precomputed SAM/OpenCLIP 512D features when RUN_SAM_CLIP=0."
fi

if [[ "$RUN_AUTOENCODER" == "1" ]]; then
    pushd autoencoder >/dev/null
    ae_dataset_path="../data/${DATASET}/dust3r_${N_VIEWS}_views"
    run "$PYTHON_BIN" train.py \
        --dataset_path "$ae_dataset_path" \
        --encoder_dims 256 128 64 32 3 \
        --decoder_dims 16 32 64 128 256 256 512 \
        --lr "$AE_LR" \
        --dataset_name "$DATASET"
    run "$PYTHON_BIN" test.py \
        --dataset_path "$ae_dataset_path" \
        --dataset_name "$DATASET"
    popd >/dev/null
elif [[ "$NEED_DIM3_FEATURES" == "1" ]]; then
    require_dir "${SOURCE_PATH}/language_features_dim3" \
        "Semantic training/rendering needs precomputed 3D language features when RUN_AUTOENCODER=0."
fi

if [[ "$RUN_LANG_FUSION" == "1" ]]; then
    require_dir "${SOURCE_PATH}/language_features" \
        "Language fusion needs SAM/OpenCLIP 512D features."
    require_dir "${SOURCE_PATH}/language_features_dim3" \
        "Language fusion needs 3D language features from autoencoder/test.py or a precomputed equivalent."
    require_file "${SOURCE_PATH}/sparse/0/intrinsics.pt" \
        "Language fusion uses sparse/0 camera intrinsics."
    require_file "${SOURCE_PATH}/sparse/0/poses.pt" \
        "Language fusion uses sparse/0 camera poses."
    require_file "${SOURCE_PATH}/sparse/0/depths.pt" \
        "Language fusion uses sparse/0 depth estimates."
    lang_fusion_cmd=(
        "$PYTHON_BIN" ./lang_fusion.py
        --dataname "$DATASET"
        --n_views "$N_VIEWS"
        --feature_level "$LANG_FUSION_LEVEL"
    )
    if [[ "$LANG_FUSION_REFRESH_ORIGIN" == "1" ]]; then
        lang_fusion_cmd+=(--refresh_origin)
    fi
    if [[ "$LANG_FUSION_VISUALIZE" != "1" ]]; then
        lang_fusion_cmd+=(--no_visualization)
    fi
    run "${lang_fusion_cmd[@]}"
fi

if [[ "$RUN_FEATURE_TRAIN" == "1" ]]; then
    for feature_level in "${FEATURE_LEVEL_ARRAY[@]}"; do
        run "$PYTHON_BIN" ./train_joint.py \
            -s "$SOURCE_PATH" \
            -m "$MODEL_ROOT" \
            --n_views "$N_VIEWS" \
            --iter "$ITER" \
            --start_checkpoint "$START_CHECKPOINT" \
            --feature_level "$feature_level" \
            --include_feature_get 1
    done
fi

if [[ "$RUN_TEST_POSE_INIT" == "1" ]]; then
    require_file "$DUST3R_CKPT" "Set DUST3R_CKPT=/path/to/checkpoint if it is elsewhere."
    run "$PYTHON_BIN" ./init_test_pose.py \
        --model_path "$DUST3R_CKPT" \
        --img_base_path "$IMAGE_PATH" \
        --n_views "$N_VIEWS" \
        --focal_avg
fi

if [[ "$RUN_RENDER" == "1" ]]; then
    for feature_level in "${FEATURE_LEVEL_ARRAY[@]}"; do
        require_file "${MODEL_ROOT}_${feature_level}/chkpnt${FEATURE_CKPT_ITER}.pth" \
            "Set FEATURE_CKPT_ITER to the checkpoint iteration used for feature rendering."
        render_cmd=(
            "$PYTHON_BIN" ./render.py
            -s "$SOURCE_PATH"
            -m "${MODEL_ROOT}_${feature_level}"
            --n_views "$N_VIEWS"
            --feature_checkpoint_iteration "$FEATURE_CKPT_ITER"
            --include_feature
        )
        if [[ "$OPTIMIZE_TEST_POSE" == "1" ]]; then
            render_cmd+=(--optimize_test_pose)
            if [[ "$OPTIM_TEST_POSE_ITER" != "0" ]]; then
                render_cmd+=(--optim_test_pose_iter "$OPTIM_TEST_POSE_ITER")
            fi
        fi
        run "${render_cmd[@]}"
    done
fi

log "Pipeline finished."
