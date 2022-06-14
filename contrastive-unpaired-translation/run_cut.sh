#! /bin/bash

source /ocean/projects/asc170022p/yanwuxu/miniconda/etc/profile.d/conda.sh
conda activate crossMoDA
  
python train.py --dataroot ../data/T12T2_resample/ --continue_train --epoch_count 16 \
--CUT_mode CUT --load_size 448 --crop_size 448 \
--batch_size 4 --num_threads 8 \
--direction BtoA \
--master_port 69288 --world_size 4 --rank 3