#!/bin/sh

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=1

### Remember to change the reserved session time ###

# No need to load modules, local miniconda installation is used

# Dataset ID to use
export DATASET_NAME=Dataset598_SampledData
export DATASET_ID=598

# Copy data to scratch storage
mkdir -p $TMPDIR/nnUNet_raw/
cp -r $HOME/nnUNet_raw/${DATASET_NAME} $TMPDIR/nnUNet_raw/${DATASET_NAME}
mkdir -p $TMPDIR/nnUNet_preprocessed/
if [ -d $HOME/nnUNet_preprocessed/${DATASET_NAME} ]; then
    echo "${DATASET_NAME} has been preprocessed."
    cp -r $HOME/nnUNet_preprocessed/${DATASET_NAME} $TMPDIR/nnUNet_preprocessed/${DATASET_NAME}
    export preprocessed=true
else
    echo "${DATASET_NAME} has not been preprocessed. I will preprocess data."
    export preprocessed=false
fi
mkdir -p $TMPDIR/nnUNet_results/

# Set up environment
export nnUNet_raw="$TMPDIR/nnUNet_raw"
export nnUNet_preprocessed="$TMPDIR/nnUNet_preprocessed"
export nnUNet_results="$TMPDIR/nnUNet_results"

# Run training script
source /home/${USER}/.bashrc
conda activate uls
if [ "$preprocessed" = false ]; then
    nnUNetv2_extract_fingerprint -d $DATASET_ID --verify_dataset_integrity
    nnUNetv2_plan_experiment -d $DATASET_ID -pl nnUNetPlannerResEncM
    cp ${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetResEncUNetMPlans.json temp.json
    jq -r --indent 4 '."configurations"."3d_fullres"."patch_size" |= [128, 256, 256]' temp.json > ${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetResEncUNetMPlans.json
    rm temp.json
    nnUNetv2_preprocess -d $DATASET_ID -plans_name nnUNetResEncUNetMPlans
    echo "Preprocessed done."
fi
echo "Training model"
nnUNetv2_train $DATASET_ID 3d_fullres all -tr nnUNetTrainer_ULS_DCTopKLoss -p nnUNetResEncUNetMPlans

# Copy results
echo "Copying results."
alias now='date +%Y-%m-%d-%H.%M'
mkdir -p $HOME/nnUNet_results/$(now) && cp -r $TMPDIR/nnUNet_results/ $HOME/nnUNet_results/$(now)