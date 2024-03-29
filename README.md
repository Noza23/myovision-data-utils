# myovision

[![LMU: Munich](https://img.shields.io/badge/LMU-Munich-009440.svg)](https://www.en.statistik.uni-muenchen.de/index.html)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Description

This is the sub-repository of the main project [myovision](https://github.com/Noza23/myovision).
It's main purpose is to provide various utilities for the data collection process of the myovision project.

## Installation

```bash
git clone git@github.com:Noza23/myovision-data-utils.git
cd myovision-data-utils
pip install -r requirements.txt
```

Repository contains following utilities:

1. `clean_masks.py`
   The module is used to remove redundant masks manually from the given image.
   It additionally provides a visualization of the masks drawn on the image to help the user decide which masks to keep.
   See: `clean_masks.sh` for more details about parameters and usage of the script.
   Simple execution with:

   ```bash
   sh clean_masks.sh
   ```

2. `merge_masks.py`
   The module is used to merge masks from different sources into a single mask.
   In our case between:

   - Masks as RLE encoded json files generated by the annotation tool
   - .roi files generated by free-hand annotation in ImageJ.

   Check `python3 merge_masks.py --help` for more details about parameters and usage.
   Simple execution with:

   ```bash
   python3 merge_masks.py --masks /path/to/json_masks \
                            --roi /path/to/roi_masks \
                            --output /path/to/output
   ```

3. `data_generator.py`
   The module takes directory where images and corresponding masks are located:

   - Cuts each **image:mask** pair into patches of the given size.
   - Performs various data augmentation configured in `data_generator.yaml`.
   - Saves resulting data prepared for training in the given output directory.

   ```diff
   - Note: In the Directory you should have following naming convention:
        images: *.png
        masks: *_mask.json
   ```

   Fill in the `data_generator.yaml` with the desired parameters and execute the script with:

   ```bash
   python3 data_generator.py
   ```
