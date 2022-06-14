#! /bin/bash

source /ocean/projects/asc170022p/yanwuxu/miniconda/etc/profile.d/conda.sh
conda activate crossMoDA

python test.py --dataroot ../data/T12T2_resample/ \
--CUT_mode CUT --load_size 448 --crop_size 448 \
--batch_size 4 --num_threads 8 \
--direction BtoA --eval --phase train \
--master_port 66666 --world_size 1 --rank 0