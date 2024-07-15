
#export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK


julia --threads 4 icml_2024.jl  $1 $2 

