import torch
from torch import nn

import numpy as np
import cv2

import cv2
import numpy as np

def filter_largest_connected_component(img):
    """
    Filters the largest connected component in the input image.

    Parameters:
        img (ndarray): Binary 2d input image.

    Returns:
        ndarray: Binary image with only the largest connected component.

    """
    num_labels, labels_im = cv2.connectedComponents(img)
    unique, counts = np.unique(labels_im, return_counts=True)
    counts = np.argmax(counts[1:]) + 1

    return labels_im == counts

def get_axes_2d(data):
    """
    Compute the major and minor axes of a 2D binary image.

    Parameters:
        data (numpy.ndarray or torch.Tensor): The input 2D binary image.

    Returns:
        major_axis (float): The length of the major axis.
        minor_axis (float): The length of the minor axis.
    """
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    data = data.squeeze()

    # Return 0 if the input image is empty
    if np.sum(data) == 0:
        return 0, 0
    
    try:
        data = filter_largest_connected_component(data.astype(np.uint8))
        img = np.ascontiguousarray(data, dtype=np.uint8)
        contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        c = max(contours, key=cv2.contourArea)

        # Centroid
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        # Ellipse
        e = cv2.fitEllipse(c)

        # Principal axis
        x1 = int(np.round(cx + e[1][1] / 2 * np.cos((e[2] + 90) * np.pi / 180.0)))
        y1 = int(np.round(cy + e[1][1] / 2 * np.sin((e[2] + 90) * np.pi / 180.0)))
        x2 = int(np.round(cx + e[1][1] / 2 * np.cos((e[2] - 90) * np.pi / 180.0)))
        y2 = int(np.round(cy + e[1][1] / 2 * np.sin((e[2] - 90) * np.pi / 180.0)))
        major_axis = np.sqrt((x1-x2)**2 + (y1-y2)**2)

        # Second principal axis
        x1 = int(np.round(cx + e[1][0] / 2 * np.cos(e[2] * np.pi / 180.0)))
        y1 = int(np.round(cy + e[1][0] / 2 * np.sin(e[2] * np.pi / 180.0)))
        x2 = int(np.round(cx + e[1][0] / 2 * np.cos((e[2] + 180) * np.pi / 180.0)))
        y2 = int(np.round(cy + e[1][0] / 2 * np.sin((e[2] + 180) * np.pi / 180.0)))
        minor_axis = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    
    except:
        # If the ellipse fitting fails, return 0; because the ellipse fitting fails when the object is too small
        major_axis = 0
        minor_axis = 0

    return major_axis, minor_axis

def get_axes_3d(data):
    """
    Compute the axes for a 3D data array.

    Args:
        data (torch.Tensor or numpy.ndarray): The input 3D data array.

    Returns:
        torch.Tensor: A tensor containing the axes for each 2D slice of the input data.

    """
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    data = data.squeeze()

    return torch.tensor([get_axes_2d(data[i,...]) for i in range(data.shape[0])])


class AxisLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        if target.ndim == pred.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        if target.ndim != 4:
            raise NotImplementedError("AxisLoss only supports 3D data")
        
        pred_label = torch.max(pred, dim=1, keepdim=True)[1]
        pred_axes = torch.stack([get_axes_3d(pred_label[i,0,...]) for i in range(pred_label.shape[0])], dim=0).to(dtype=pred.dtype, device=pred.device)
        target_axes = torch.stack([get_axes_3d(target[i,...]) for i in range(target.shape[0])], dim=0).to(dtype=pred.dtype, device=pred.device)
        smape = (pred_axes - target_axes).abs() / (pred_axes.abs() + target_axes.abs()).clamp(min=1e-6)
        smape = torch.nan_to_num(smape) # Replace NaNs with 0 for numerical stability

        return smape.mean()


if __name__ == '__main__':
    import nibabel as nib
    import matplotlib.pyplot as plt
    file_path = "/home/godilia/nnUNet_raw/Dataset999_SmallDatasetForTest/labelsTr/000083_07_01_114_lesion_01.nii.gz"
    img = nib.load(file_path)
    data = torch.tensor(img.get_fdata(), dtype=torch.float32).squeeze().permute(2, 0, 1).view(1, 1, 128, 256, 256)
    plt.imshow(data[0,0,64,...])
    print(data.shape)
    # data_c1 = torch.empty(3, 1, 128, 256, 256).uniform_(0, 1)
    # data_c2 = 1 - data_c1
    # data = torch.cat((data_c1, data_c2), dim=1)
    # pred_label = torch.max(data, dim=1, keepdim=True)[1]
    # target = torch.empty(3, 128, 256, 256).random_(0, 2)
    target = torch.zeros(1, 1, 128, 256, 256)
    axis_loss = AxisLoss()
    print(axis_loss(data, target))