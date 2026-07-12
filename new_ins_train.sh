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
    "INITIALIZER": "INITIALIZER",
    "initializer": "INITIALIZER",
}
for key, name in top_level.items():
    if key in data:
        put(name, data[key])

stage_names = {
    "geometry_init": "RUN_GEOMETRY_INIT",
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
    "mast3r_ckpt": "MAST3R_CKPT",
    "vggt_ckpt": "VGGT_CKPT",
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
    "feature_levels": "LANG_FUSION_LEVELS",
    "refresh_origin": "LANG_FUSION_REFRESH_ORIGIN",
    "visualize": "LANG_FUSION_VISUALIZE",
    "match_threshold": "LANG_FUSION_MATCH_THRESHOLD",
    "reproject_threshold": "LANG_FUSION_REPROJECT_THRESHOLD",
    "semantic_weight": "LANG_FUSION_SEMANTIC_WEIGHT",
    "fusion_area_threshold": "LANG_FUSION_AREA_THRESHOLD",
    "certainty_threshold": "LANG_FUSION_CERTAINTY_THRESHOLD",
    "min_roma_pixels": "LANG_FUSION_MIN_ROMA_PIXELS",
    "min_reproject_pixels": "LANG_FUSION_MIN_REPROJECT_PIXELS",
    "mask_fusion": "LANG_FUSION_MASK_FUSION",
    "similarity_space": "LANG_FUSION_SIMILARITY_SPACE",
    "roma_similarity_space": "LANG_FUSION_ROMA_SIMILARITY_SPACE",
    "reproject_similarity_space": "LANG_FUSION_REPROJECT_SIMILARITY_SPACE",
    "scoring": "LANG_FUSION_SCORING",
    "matcher": "LANG_FUSION_MATCHER",
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
INITIALIZER="${INITIALIZER:-dust3r}"
ITER="${ITER:-1000}"
BASE_FEATURE_LEVEL="${BASE_FEATURE_LEVEL:-3}"
FEATURE_LEVELS="${FEATURE_LEVELS:-2}"

# Stages. Defaults keep the old workflow: train geometry, make SAM/CLIP
# features, train semantic Gaussians, and render. DUSt3R init and autoencoder
# compression are optional because many runs reuse precomputed outputs.
RUN_GEOMETRY_INIT="${RUN_GEOMETRY_INIT:-${RUN_DUST3R_INIT:-0}}"
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
MAST3R_CKPT="${MAST3R_CKPT:-./ckpt/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
VGGT_CKPT="${VGGT_CKPT:-./ckpt/model.pt}"
SAM_CKPT="${SAM_CKPT:-./ckpt/sam_vit_h_4b8939.pth}"
AE_LR="${AE_LR:-0.0007}"
LANG_FUSION_LEVEL="${LANG_FUSION_LEVEL:-}"
LANG_FUSION_LEVELS="${LANG_FUSION_LEVELS:-}"
LANG_FUSION_REFRESH_ORIGIN="${LANG_FUSION_REFRESH_ORIGIN:-0}"
LANG_FUSION_VISUALIZE="${LANG_FUSION_VISUALIZE:-0}"
LANG_FUSION_MATCH_THRESHOLD="${LANG_FUSION_MATCH_THRESHOLD:-}"
LANG_FUSION_REPROJECT_THRESHOLD="${LANG_FUSION_REPROJECT_THRESHOLD:-}"
LANG_FUSION_SEMANTIC_WEIGHT="${LANG_FUSION_SEMANTIC_WEIGHT:-}"
LANG_FUSION_AREA_THRESHOLD="${LANG_FUSION_AREA_THRESHOLD:-}"
LANG_FUSION_CERTAINTY_THRESHOLD="${LANG_FUSION_CERTAINTY_THRESHOLD:-}"
LANG_FUSION_MIN_ROMA_PIXELS="${LANG_FUSION_MIN_ROMA_PIXELS:-}"
LANG_FUSION_MIN_REPROJECT_PIXELS="${LANG_FUSION_MIN_REPROJECT_PIXELS:-}"
LANG_FUSION_MASK_FUSION="${LANG_FUSION_MASK_FUSION:-}"
LANG_FUSION_SIMILARITY_SPACE="${LANG_FUSION_SIMILARITY_SPACE:-}"
LANG_FUSION_ROMA_SIMILARITY_SPACE="${LANG_FUSION_ROMA_SIMILARITY_SPACE:-}"
LANG_FUSION_REPROJECT_SIMILARITY_SPACE="${LANG_FUSION_REPROJECT_SIMILARITY_SPACE:-}"
LANG_FUSION_SCORING="${LANG_FUSION_SCORING:-}"
LANG_FUSION_MATCHER="${LANG_FUSION_MATCHER:-}"
LANG_FUSION_STATS_PATH="${LANG_FUSION_STATS_PATH:-}"
LANG_FUSION_SNAPSHOT_DIR="${LANG_FUSION_SNAPSHOT_DIR:-}"
OPTIMIZE_TEST_POSE="${OPTIMIZE_TEST_POSE:-0}"
OPTIM_TEST_POSE_ITER="${OPTIM_TEST_POSE_ITER:-0}"
FEATURE_CKPT_ITER="${FEATURE_CKPT_ITER:-1000}"
USER_START_CHECKPOINT="${START_CHECKPOINT:-}"
MAX_INIT_POINTS="${MAX_INIT_POINTS:-0}"
INIT_POINT_SEED="${INIT_POINT_SEED:-0}"
MAX_INIT_SCALE="${MAX_INIT_SCALE:-0}"
TRAIN_SEED="${TRAIN_SEED:-}"
RESET_START_ITERATION="${RESET_START_ITERATION:-1}"

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
LANG_FUSION_LEVELS="${LANG_FUSION_LEVELS:-${LANG_FUSION_LEVEL}}"

IMAGE_PATH="./data/${DATASET}"
SOURCE_PATH="${IMAGE_PATH}/dust3r_${N_VIEWS}_views"
MODEL_ROOT="./output/${DATASET}/${N_VIEWS}_views"
BASE_MODEL_PATH="${MODEL_ROOT}_${BASE_FEATURE_LEVEL}"
SOURCE_GEOMETRY_DIR="${SOURCE_PATH}/dust3r_${N_VIEWS}_views"
BASE_CKPT="${BASE_MODEL_PATH}/chkpnt${ITER}.pth"
COPIED_CKPT="${SOURCE_GEOMETRY_DIR}/chkpnt${ITER}.pth"

log "Dataset=${DATASET}, N_VIEWS=${N_VIEWS}, ITER=${ITER}"
log "Initializer=${INITIALIZER}"
log "Source=${SOURCE_PATH}"
log "Model root=${MODEL_ROOT}"
log "Feature levels=${FEATURE_LEVELS}"
if [[ "$RUN_LANG_FUSION" == "1" ]]; then
    log "Language fusion levels=${LANG_FUSION_LEVELS}"
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

if [[ "$RUN_GEOMETRY_INIT" == "1" ]]; then
    require_dir "${IMAGE_PATH}/images" "Geometry initialization samples sparse views from this folder."
    case "$INITIALIZER" in
        dust3r)
            require_file "$DUST3R_CKPT" "Set DUST3R_CKPT if the checkpoint is elsewhere."
            run "$PYTHON_BIN" ./coarse_init_eval.py \
                --model_path "$DUST3R_CKPT" --n_views "$N_VIEWS" \
                --img_base_path "$IMAGE_PATH" --focal_avg
            ;;
        mast3r)
            require_dir ./mast3r "Clone the official MASt3R repository here; see README.md."
            require_file "$MAST3R_CKPT" "Set MAST3R_CKPT if the checkpoint is elsewhere."
            run "$PYTHON_BIN" ./coarse_init_mast3r_eval.py \
                --model_path "$MAST3R_CKPT" --n_views "$N_VIEWS" \
                --img_base_path "$IMAGE_PATH"
            ;;
        vggt)
            require_dir ./vggt "Clone the official VGGT repository here; see README.md."
            require_file "$VGGT_CKPT" "Set VGGT_CKPT if the checkpoint is elsewhere."
            run "$PYTHON_BIN" ./coarse_init_vggt_eval.py \
                --model_path "$VGGT_CKPT" --n_views "$N_VIEWS" \
                --img_base_path "$IMAGE_PATH"
            ;;
        *) die "INITIALIZER must be one of: dust3r, mast3r, vggt" ;;
    esac
else
    require_dir "${SOURCE_PATH}/images" \
        "Run with RUN_GEOMETRY_INIT=1, or prepare the sparse-view image folder first."
    require_dir "${SOURCE_PATH}/sparse/0" \
        "Run with RUN_GEOMETRY_INIT=1, or prepare initializer/COLMAP files first."
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
        --max_init_points "$MAX_INIT_POINTS"
        --init_point_seed "$INIT_POINT_SEED"
        --max_init_scale "$MAX_INIT_SCALE"
    )
    if [[ -n "$TRAIN_SEED" ]]; then
        rgb_train_cmd+=(--seed "$TRAIN_SEED")
    fi
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
        --feature_levels "$LANG_FUSION_LEVELS"
    )
    if [[ "$LANG_FUSION_REFRESH_ORIGIN" == "1" ]]; then
        lang_fusion_cmd+=(--refresh_origin)
    fi
    if [[ "$LANG_FUSION_VISUALIZE" != "1" ]]; then
        lang_fusion_cmd+=(--no_visualization)
    fi
    if [[ -n "$LANG_FUSION_MATCH_THRESHOLD" ]]; then
        lang_fusion_cmd+=(--match_threshold "$LANG_FUSION_MATCH_THRESHOLD")
    fi
    if [[ -n "$LANG_FUSION_REPROJECT_THRESHOLD" ]]; then
        lang_fusion_cmd+=(--reproject_threshold "$LANG_FUSION_REPROJECT_THRESHOLD")
    fi
    if [[ -n "$LANG_FUSION_SEMANTIC_WEIGHT" ]]; then
        lang_fusion_cmd+=(--semantic_weight "$LANG_FUSION_SEMANTIC_WEIGHT")
    fi
    if [[ -n "$LANG_FUSION_AREA_THRESHOLD" ]]; then
        lang_fusion_cmd+=(--fusion_area_threshold "$LANG_FUSION_AREA_THRESHOLD")
    fi
    if [[ -n "$LANG_FUSION_CERTAINTY_THRESHOLD" ]]; then
        lang_fusion_cmd+=(--certainty_threshold "$LANG_FUSION_CERTAINTY_THRESHOLD")
    fi
    if [[ -n "$LANG_FUSION_MIN_ROMA_PIXELS" ]]; then
        lang_fusion_cmd+=(--min_roma_pixels "$LANG_FUSION_MIN_ROMA_PIXELS")
    fi
    if [[ -n "$LANG_FUSION_MIN_REPROJECT_PIXELS" ]]; then
        lang_fusion_cmd+=(--min_reproject_pixels "$LANG_FUSION_MIN_REPROJECT_PIXELS")
    fi
    if [[ -n "$LANG_FUSION_MASK_FUSION" ]]; then
        lang_fusion_cmd+=(--mask_fusion "$LANG_FUSION_MASK_FUSION")
    fi
    if [[ -n "$LANG_FUSION_SIMILARITY_SPACE" ]]; then
        lang_fusion_cmd+=(--similarity_space "$LANG_FUSION_SIMILARITY_SPACE")
    fi
    if [[ -n "$LANG_FUSION_ROMA_SIMILARITY_SPACE" ]]; then
        lang_fusion_cmd+=(--roma_similarity_space "$LANG_FUSION_ROMA_SIMILARITY_SPACE")
    fi
    if [[ -n "$LANG_FUSION_REPROJECT_SIMILARITY_SPACE" ]]; then
        lang_fusion_cmd+=(--reproject_similarity_space "$LANG_FUSION_REPROJECT_SIMILARITY_SPACE")
    fi
    if [[ -n "$LANG_FUSION_SCORING" ]]; then
        lang_fusion_cmd+=(--scoring "$LANG_FUSION_SCORING")
    fi
    if [[ -n "$LANG_FUSION_MATCHER" ]]; then
        lang_fusion_cmd+=(--matcher "$LANG_FUSION_MATCHER")
    fi
    if [[ -n "$LANG_FUSION_STATS_PATH" ]]; then
        lang_fusion_cmd+=(--stats_path "$LANG_FUSION_STATS_PATH")
    fi
    if [[ -n "$LANG_FUSION_SNAPSHOT_DIR" ]]; then
        lang_fusion_cmd+=(--snapshot_dir "$LANG_FUSION_SNAPSHOT_DIR")
    fi
    run "${lang_fusion_cmd[@]}"
fi

if [[ "$RUN_FEATURE_TRAIN" == "1" ]]; then
    for feature_level in "${FEATURE_LEVEL_ARRAY[@]}"; do
        feature_train_cmd=(
            "$PYTHON_BIN" ./train_joint.py
            -s "$SOURCE_PATH" \
            -m "$MODEL_ROOT" \
            --n_views "$N_VIEWS" \
            --iter "$ITER" \
            --start_checkpoint "$START_CHECKPOINT" \
            --feature_level "$feature_level" \
            --include_feature_get 1
        )
        if [[ "$RESET_START_ITERATION" == "1" ]]; then
            feature_train_cmd+=(--reset_start_iteration)
        fi
        if [[ -n "$TRAIN_SEED" ]]; then
            feature_train_cmd+=(--seed "$TRAIN_SEED")
        fi
        run "${feature_train_cmd[@]}"
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
