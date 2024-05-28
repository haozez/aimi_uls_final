#!/bin/bash
#SBATCH --account=cseduimc070
#SBATCH --partition=csedu-prio,csedu
#SBATCH --qos=csedu-preempt
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=myjob-%j.out
#SBATCH --error=myjob-%j.err

### Remember to change the reserved session time ###

# No need to load modules, local miniconda installation is used

# Dataset ID to use
export DATASET_NAME=Dataset598_SampledData
export DATASET_ID=598

# Copy data to scratch storage
echo "Copying data to $TMPDIR"
mkdir -p $TMPDIR/nnUNet_raw/
cp -r $HOME/nnUNet_raw/${DATASET_NAME} $TMPDIR/nnUNet_raw/${DATASET_NAME}
mkdir -p $TMPDIR/nnUNet_preprocessed/
if [ -d $HOME/nnUNet_preprocessed/${DATASET_NAME} ]; then
    echo "${DATASET_NAME} has been preprocessed. Copying preprocessed data to $TMPDIR"
    cp -r $HOME/nnUNet_preprocessed/${DATASET_NAME} $TMPDIR/nnUNet_preprocessed/${DATASET_NAME}
    export preprocessed=true
else
    echo "${DATASET_NAME} has not been preprocessed. I will preprocess data."
    export preprocessed=false
fi
mkdir -p $TMPDIR/nnUNet_results/

# Set up environment
export nnUNet_raw="$TMPDIR/nnUNet_raw/"
export nnUNet_preprocessed="$TMPDIR/nnUNet_preprocessed/"
export nnUNet_results="$TMPDIR/nnUNet_results/"

# Run training script
source /home/${USER}/.bashrc
conda activate uls
if [ "$preprocessed" = false ]; then
    nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity -pl nnUNetPlannerResEncM
    cp -r $TMPDIR/nnUNet_preprocessed/${DATASET_NAME} $HOME/nnUNet_preprocessed/${DATASET_NAME}
    echo "Preprocessed done. You need to check the plan.json file and modify it if necessary."
    echo "Exiting..."
    exit 0
fi
echo "Training model"
nnUNetv2_train $DATASET_ID 3d_fullres all -tr nnUNetTrainer_ULS_DCFocalLoss -p nnUNetResEncUNetMPlans

# Copy results back to home
echo "Copying results back to $HOME"
alias now='date +%Y-%m-%d-%H.%M'
mkdir -p $HOME/nnUNet_results/$(now) && cp -r $TMPDIR/nnUNet_results/ $HOME/nnUNet_results/$(now)