### AIMI Team 6, ULS23 Challenge 
Team members: O. Geertsema, D. Santos, V. Petre and H. Zhu

Supervisor: A. Hering

### Introduction

This repository contains the code for the AIMI Team 6 (Radboud University) in the ULS23 Challenge. The challenge is to segment lesions in medical images. More information about the challenge can be found on the [ULS23 Challenge website](https://uls23.grand-challenge.org/).

We followed the nnUNet pipeline to train our models. For general information for preprocessing and training, please refer to the [nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet?tab=readme-ov-file#how-to-get-started).

Folder structure:
- `extensions/nnunetv2`: Our main contributions, including new loss functions with complementary experiment planners and network trainers.
- `misc_scripts`: Miscellaneous scripts, including data preprocessing, data sampling, and job scripts.
- `nnUNet`: The full nnUNet package with our modifications added; for replication purposes.

### Our main contributions: experimental with new loss functions

We experimented with multiple losses function to address the class imbalance issue and increase the robustness of the model towards input perturbations. Please notice that we used the latest ResNet version of nnUNet; is you do not intend to use the ResNet version, please remove `-p nnUNetResEncUNetMPlans` from the command line. Our changes can be found in the `extensions/nnunetv2` folder.

#### Focal loss and Top-k loss
We used the Focal loss and Top-k loss to address the class imbalance issue. The Focal loss is a modification of the cross-entropy loss that down-weights the loss assigned to well-classified examples. The Top-k loss is another modification of the cross-entropy loss that only considers the k most probable classes. We set k to 10 in our experiments.

To train the nnUNet with Focal loss for 500 epochs ($lr = 0.01$), use the following command:
```
nnUNetv2_train DATASET CONFIG FOLD -tr nnUNetTrainer_ULS_DCFocalLoss -p nnUNetResEncUNetMPlans
```
in which `DATASET` is the dataset ID or name, `CONFIG` is the configuration (`2d` or `3d_fullres`) and `FOLD` is the fold number for training (or `all` if not using cross-validation).

To train the nnUNet with Top-k loss for 500 epochs ($lr = 0.01$), use the following command:
```
nnUNetv2_train DATASET CONFIG FOLD -tr nnUNetTrainer_ULS_DCTopKLoss -p nnUNetResEncUNetMPlans
```

#### Long- and short-axis matching loss
The ULS task is not trivial, and we sub-selected part of the data to train the mode due to time constrain (data sampling strategy will be described later); we hypothesized that the model could benefit from a loss function that captures global information about lesions. Therefore, we implemented a loss function considering whether the long- and short-axis of the lesions match between prediction and target. The pseudo-code of the loss function is as follows:
```
Def get_axis(3d_binary_image):
    For each slice along the z-axis do:
	    If no positive label in the slice:
            Return long_axis = 0, short_axis = 0
        Else:
            Find the largest connected component in the slice
            Fit an ellipse to that largest connected component
            Return the long_axis and short_axis of that ellipse

Def long_short_axis_loss(nn.Module):
    Pred_labels = binarize the network output
    Pred_long_axis, pred_short_axis = get_axis(pred_labels)
    Target_long_axis, target_short_axis = get_axis(target)
    Return SMAPE(pred_long_axis, target_long_axis) + SMAPE(pred_short_axis, target_short_axis)
```

We combined the long- and short-axis matching loss with the cross-entropy loss and Dice loss. The ratio between these losses are $4:4:2$ (CE : Dice : axis_loss).

To train the nnUNet with long- and short-axis matching loss for 500 epochs ($lr = 0.01$), use the following command:
```
nnUNetv2_train DATASET CONFIG FOLD -tr nnUNetTrainer_ULS_DCCEAxisLoss -p nnUNetResEncUNetMPlans
```

#### (Rotation) robustness loss

We also implemented a loss function to increase the robustness of the model towards input perturbations. For this, we rotated the input images by 180 trained the model to predict the same segmentation mask for both the original and rotated images. We combined the rotation robustness loss with the cross-entropy loss and Dice loss. The ratio between these losses are $4:4:2$ (CE : Dice : robust_loss).

To train the nnUNet with rotation robustness loss for 500 epochs ($lr = 0.01$), use the following command:
```
nnUNetv2_train DATASET CONFIG FOLD -tr nnUNetTrainer_ULS_DCCEAxisLoss -p nnUNetResEncUNetMPlans
```

### Miscellaneous technical details

Beyond the loss functions, we also implemented a few other functionality to improve the model performance. We used the following data augmentation techniques: random rotation, random scaling, random elastic deformation, random flipping, and random intensity shift. We also used the following post-processing techniques: thresholding, connected component analysis, and morphological operations.

#### Data preprocessing

Although the published dataset has been preprocessed by the organizers in terms of patchification and normalization, we further preprocessed the dataset to match the nnUNet format. We did the following preprocessing steps:
- The dataset were published as split zip archives (i.e., *.zip, *.z01, *.z02, etc.). We merged the split archives and unzipped the dataset files.
- Some of the images do not come with a label file. We detected and removed these images from the dataset.
- We added a 4-digit dummy channel identifier to the enf of the image and file names, per the requirement of nnUNet pipeline.
- We create a `dataset.json` file that contains the dataset information, such as the channel names, the label names, and the number of training samples. The `dataset.json` file template is available in the `misc_scripts` folder; please modify the template to match the dataset (number of samples).
- We made the preprocessed dataset available in the project space on Snellius, and we are aware that other teams have taken advantage of this preprocessed dataset.

#### Data sampling strategy

We used a data sampling strategy to train the model. We randomly selected 10% of the data from each dataset for training, and 2.5% for validation. We used the same data sampling strategy for all experiments. For two datasets (`DSC_Task06_Lung`, `MDSC_Task10_Colon`) which less than 100 samples, we 80% of the data for training and 20% for validation, which ensures that these lesions are well represented in the training and validation sets.

The data sampling script is available in the `misc_scripts` folder.

#### Fix image head mismatching between image and label

We noticed that, for a few data samples, the image and label headers were mismatched in terms of significant digits. We detected this issue by comparing the image and label headers and fixed it by using the image header to replace the label header. We have carefully checked that the mismatching issue only comes from the differences of significant digits, and the image and label headers are actually consistent.

#### Adjust patch size for 3D models

The nnUNet pipeline automatically adjusts the patch size during planning. However, our dataset has been preprocessed to make sure that one 3D patch contains the entire lesion. Automatically adjusting the patch size may result in a patch that does not contain the entire lesion. Therefore, we manually set the patch size to 256x256x128 (x, y, z) for 3D models. We achieved this by modifying the patch size in the plan file before running preprocessing.

#### Training scripts

We used a job script to train the models on the Snellius cluster. The job script is available in the `misc_scripts` folder. In the script, we first check if nnUNet preprocessed the dataset, and if not, we preprocess the dataset first. Then we train the model using the nnUNet pipeline.

The job script is a template, and you need to modify the script to match your dataset and configuration. Meanwhile, the job script is designed to run on the Snellius cluster, and you may need to modify the script to run on other clusters.

#### Trained model weights

The trained model weights are available per request. We unfortunately cannot provide the weights directly in the repository due to the size limitation of the repository.