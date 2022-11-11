#!/bin/bash
#SBATCH -N 1
#SBATCH -c 30
#SBATCH --mem=0
#SBATCH -J spm-big
#SBATCH -A venkvis
#SBATCH -p highmem
#SBATCH -t 1-00:00

spack load julia@1.8
julia --project=. scripts/jl/spm_pack_big.jl
