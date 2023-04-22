#!/bin/bash
module load learning/conda-5.1.0-py36-gpu && source activate deep_packet
module load cuda/11.8.0 && module load cudnn
echo "Setup Done"
