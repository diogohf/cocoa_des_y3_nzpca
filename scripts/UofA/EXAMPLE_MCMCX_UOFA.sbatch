#!/bin/bash

#SBATCH --job-name=LSSTY1MC
#SBATCH --output=LSSTY1MC-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --cpus-per-task=10
#SBATCH --time=100:00:00
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=timeifler

##SBATCH --partition=standard
##SBATCH --account=cosmolike

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
source ~/.bashrc 
conda init bash
source ~/.bashrc 

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of MPI cores is $SLURM_NTASKS
echo Number of OpenMP cores is $SLURM_CPUS_PER_TASK
echo SLURM DIR is $SLURM_SUBMIT_DIR

cd $SLURM_SUBMIT_DIR
conda activate cocoa
source start_cocoa

echo Cocoa Conda environment Activated

export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

mpirun -n ${SLURM_NTASKS} --oversubscribe --mca btl tcp,self --bind-to core:overload-allowed --map-by node:pe=${OMP_NUM_THREADS} cobaya-run ./projects/des_y3/EXAMPLE_MCMC_${SLURM_ARRAY_TASK_ID}.yaml -r