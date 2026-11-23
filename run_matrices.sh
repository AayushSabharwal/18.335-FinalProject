#!/bin/bash

module load apptainer
apptainer pull $HOME/images/freefem.sif docker://freefem/freefem:latest
sbatch matrices.sh
