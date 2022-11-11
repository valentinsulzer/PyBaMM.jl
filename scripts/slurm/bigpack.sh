#!/bin/bash
#SBATCH -N 1
#SBATCH -c 30
#SBATCH --mem=0
#SBATCH -J bigpack
#SBATCH -A venkvis
#SBATCH -p highmem
#SBATCH -t 5-00:00

spack load julia@1.8
julia --project=. scripts/jl/run_big_pack.jl
