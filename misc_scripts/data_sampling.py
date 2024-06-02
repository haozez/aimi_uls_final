import numpy as np
import os
import shutil
import re
import json
from pathlib import Path
import nibabel as nib


# The possible extensions of data, used to filter out
# unwanted junk
file_extensions = (".nii.gz", ".nii.gz.zip")

# The names of the datasets which are too small to use a 
# subset of. Directory has to be specified in the way shown
# here for the script to work
to_ignore = ['Dataset506_MDSC_Task06_Lung/imagesTr', 
             'Dataset508_MDSC_Task10_Colon/imagesTr']


def extract_patient_id(filename):
    ''' Use regular expression to extract
    patient ID from filenames
    '''
    # Define the regular expression pattern
    pattern = r".*?_(\d+)_.*?\.nii"

    # Search for the pattern in the filename
    match = re.search(pattern, filename)

    if match:
        # Extract the patient ID from the matched group
        patient_id = match.group(1)
        return patient_id
    else:
        return None
        
def remove_extensions(file_name):
    """
    Remove all extensions from a file name.
    """
    base_name, extension = os.path.splitext(file_name)
    while extension:
        base_name, extension = os.path.splitext(base_name)
    return base_name

def check_and_correct_header(image_path, label_path):
    ''' Corrects the header of the nifti file to make sure it is
    in the correct numerical decimal. '''
    image = nib.load(image_path)
    label = nib.load(label_path)
    if image.header != label.header:
        print(f'Header mismatch for {image_path} and {label_path}. Correcting...')
        nib.save(nib.Nifti1Image(label.get_fdata().copy(), header=image.header, affine=image.affine), label_path)

def create_splits(source_folder, seed=3):
    ''' Creates splits of datapoints. If the dataset is marked as to_ignore,
    the entire dataset is split into 80% and 20% training and validation splits.
    If not marked as to_ignore, 10% of the dataset is used as a training split,
    and 2.5% as a validation split. This uses 12.5% of the total data, 80% of
    which is for training and the rest for validation.
    
    Patient IDs do not overlap across the generated training and validation 
    splits.

    Not all random seeds work; try a different seed if the previous fails.
    
    '''

    # Set random seed for replicability
    np.random.seed(seed)

    # Get the list of files in the source folder
    files = os.listdir(source_folder)

    # Filter files based on file extensions, avoid .DS_Store and other junk
    filtered_files = np.array([file for file in files 
                      if any(file.endswith(ext) for ext in file_extensions)])
    
    # Create randomly-selected data subsets for training + validation, 
    # unless the dataset is below a certain size, in which case the entire 
    # dataset is split into training and validation splits
    
    if not source_folder in to_ignore:

        # 10% training split
        training_split = np.random.choice(
            filtered_files, size=len(filtered_files)//10, replace=False)
        # Keep track of patient IDs allocated to the split to make sure they
        # don't overlap with the validation
        training_IDs = []
        for file in training_split:
            patient_ID = extract_patient_id(file)
            assert patient_ID != None, 'Patient ID not found'
            training_IDs.append(patient_ID)

        # 2.5% validation split 
        # (= 12.5% of data used total, 80% for training, 20% validation)

        shuffled_files = np.random.choice(
            filtered_files, size=len(filtered_files), replace=False)
        validation_split = []
        validation_IDs = []
        for file in shuffled_files:
            patient_ID = extract_patient_id(file)
            assert patient_ID != None, 'Patient ID not found'

            if not file in training_split and not patient_ID in training_IDs:
                validation_split.append(file)
                validation_IDs.append(patient_ID)
            
            if len(validation_split) >= len(filtered_files)//40:
                break

        validation_split = np.array(validation_split)

        # Checks to make sure things worked properly
        assert len(np.unique(training_split)) == len(training_split), \
            'Duplicates in training split'
        assert len(training_split) >= len(filtered_files)//10, \
            'Insufficient files allocated to training split'
        
        assert len(np.unique(validation_split)) == len(validation_split), \
            'Duplicates in validation split'
        assert len(validation_split) >= len(filtered_files)//40, \
            'Insufficient files allocated to validation split'

        common_elements = np.intersect1d(validation_split, training_split)
        assert len(common_elements) == 0, 'Datapoints shared between splits'

        return training_split, validation_split

    else:
        # Go by IDs; include all of a patient's scans to the validation set, 
        # patient by patient, until the 20% threshold is exceeded. The rest
        # becomes the training set.
        shuffled_files = np.random.choice(
            filtered_files, size=len(filtered_files), replace=False)
        
        all_lesion_IDs = np.array(
            [extract_patient_id(filename) for filename in shuffled_files])

        validation_split = [] 
        validation_IDs = []
        for ID in np.unique(all_lesion_IDs):
            patient_lesions = list(np.squeeze(
                shuffled_files[np.argwhere(all_lesion_IDs == ID)],axis=-1))
            for lesion in patient_lesions:
                validation_split.append(lesion)
                validation_IDs.append(ID)
            if len(validation_split) >= len(shuffled_files)//5:
                break

        validation_split = np.array(validation_split)

        training_split = np.array([file for file in shuffled_files if \
                                   file not in validation_split])

                
        # Checks to make sure things worked properly
        assert len(np.unique(training_split)) == len(training_split), \
            'Duplicates in training split'
        assert len(np.unique(validation_split)) == len(validation_split), \
            'Duplicates in validation split'
        
        assert len(validation_split) >= len(filtered_files)//5, \
            'Insufficient files allocated to validation split'     

        common_elements = np.intersect1d(validation_split, training_split)
        assert len(common_elements) == 0, 'Datapoints shared between splits'

        return training_split, validation_split


def process_entire_dataset():
    ''' Processes each of the datasets within the complete dataset. Generates
    a new dataset folder in the parent directory, following the same file structure 
    as the original complete dataset. Also modifies the .json files for each
    of the separate datasets. '''

    # set the directory to the parent folder of the datasets
    os.chdir('/projects/0/nwo2021061/uls23/nnUNet_raw')
 
    # Get the list of files in the source folder
    datasets = os.listdir()

    # Filter files only, avoid .DS_Store and other junk
    # These are the main dataset folders
    datasets = np.array([file for file in datasets 
                      if os.path.isdir(file)])
    
    # Create the parent folder for the resulting dataset
    processed_data_dir = 'Dataset598_SampledData'
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
        os.makedirs(os.path.join(processed_data_dir, 'imagesTr'))
        os.makedirs(os.path.join(processed_data_dir, 'labelsTr'))  



    complete_training_split = []
    complete_validation_split = []

    for dataset_name in datasets:
        if dataset_name == 'Dataset599_SampledDataWithTest' or dataset_name == 'Dataset598_SampledData':
            continue
        print(f'\nProcessing dataset {dataset_name}')
        training_images_path = os.path.join(dataset_name, 'imagesTr')

        # These are the same file names for both images and labels, can reuse
        training_split, validation_split = create_splits(training_images_path)

        complete_training_split += list(training_split)
        complete_validation_split += list(validation_split)

        # Training split
        for file in training_split:
            # Images
            image_source_path = os.path.join(dataset_name, 'imagesTr', file)
            image_target_path = os.path.join(processed_data_dir, 'imagesTr', file)
            shutil.copy2(image_source_path, image_target_path)
            # Labels
            label_file = file.split(os.extsep)[0][:-5] + '.' + file.split(os.extsep, 1)[1]
            label_source_path = os.path.join(dataset_name, 'labelsTr', label_file)
            label_target_path = os.path.join(processed_data_dir, 'labelsTr', label_file)
            shutil.copy2(label_source_path, label_target_path)
            # Correct the header of the label file to match the image file
            check_and_correct_header(image_target_path, label_target_path)
        
        # Validation split. Still put this in the imagesTr and labelsTr files, but
        # will indicate that these are for validation use in the specific .json file
        
        for file in validation_split:
            # Images
            image_source_path = os.path.join(dataset_name, 'imagesTr', file)
            image_target_path = os.path.join(processed_data_dir, 'imagesTr', file)
            shutil.copy2(image_source_path, image_target_path)
            # Labels
            label_file = file.split(os.extsep)[0][:-5] + '.' + file.split(os.extsep, 1)[1]
            label_source_path = os.path.join(dataset_name, 'labelsTr', label_file)
            label_target_path = os.path.join(processed_data_dir, 'labelsTr', label_file)
            shutil.copy2(label_source_path, label_target_path)  
            # Correct the header of the label file to match the image file
            check_and_correct_header(image_target_path, label_target_path)     


    # Create the split json file. One-fold validation.
    
    # Need to process the file names to remove extensions and channel names
    for i in range(len(complete_training_split)):
        complete_training_split[i] = remove_extensions(complete_training_split[i])[:-5]
        
    for i in range(len(complete_validation_split)):
        complete_validation_split[i] = remove_extensions(complete_validation_split[i])[:-5]
    
    data_splits = [{'train': complete_training_split, 'val': complete_validation_split}]
    with open(os.path.join(processed_data_dir, 'splits_final.json'), 'w') as file:
        data = json.dump(data_splits, file, indent=1)

    # Create the dataset.json file, by copying from some random dataset folder       
    shutil.copy2(os.path.join(datasets[0], 'dataset.json'), 
        os.path.join(processed_data_dir, 'dataset.json'))    

    # Modify to include the correct number of training datapoints after
    # copying over, and the appropriate channel label
    with open(os.path.join(processed_data_dir, 'dataset.json'), 'r') as file:
        data = json.load(file)

    data['numTraining'] = len(complete_training_split) + len(complete_validation_split) # Should count both tr and val data
    data['channel_names']['0'] = 'CT'

    with open(os.path.join(processed_data_dir, 'dataset.json'), 'w') as file:
        json.dump(data, file, indent=1) 

if __name__ == "__main__":
    process_entire_dataset()

