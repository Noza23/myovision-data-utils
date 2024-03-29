#!/bin/bash

python3 clean_masks.py \
    --image data_in/img.tif \
    --masks data_in/masks.json \
    --gamma 0.3 \
    --omit_cluster_values \
    --clahe 1 \
    --invert 1 \
    --delete \
    --output data_in/done \



# On windows:
# python3 data_correction.py --image path_to_image --masks path_to_json_mask --gamma 1 --omit_cluster_values --clahe 0 --invert 0 --delete --output path

# Usage:
# --image: path to image
# --masks: path to masks json file
# --gamma: gamma correction (default: 1)
# --omit_cluster_values: omit list of clusters, list clusters with space separated numbers (e.g: 1 2 3)
# --clahe: histogram equalization(0 - no, 1 - yes) (default: 0)
# --invert: invert image (0 - no, 1 - yes) (default: 0)
# --delete: delete masks by ids. list ids with space separated numbers (e.g: 50 12 3 10)
# --output: output directory
