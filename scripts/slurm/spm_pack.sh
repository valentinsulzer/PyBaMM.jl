#!/bin/bash
#SBATCH -N 1
#SBATCH -c 54
#SBATCH --mem=0
#SBATCH -J spm
#SBATCH -A venkvis
#SBATCH -p cpu
#SBATCH -t 1-00:00

spack load julia@1.8
julia --project=. scripts/jl/spm_pack.jl
