#!/bin/bash

# Correctly initialize conda for non-interactive shell
source ~/miniconda3/etc/profile.d/conda.sh

conda activate FTS_download

python /home/vincent-1080/repo/Finance_data/data_crawl.py