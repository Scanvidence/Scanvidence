"""3D Region of Interest (ROI) extraction from segmentation masks."""

import numpy as np


def extract_roi_bounding_box(segmentation_mask: np.ndarray, padding_pct: float = 0.10):
    """Extract 3D bounding box coordinates from a binary or multi-class segmentation mask.

    This function is a critical component of the Two-Stream Pipeline. It takes
    the output of the T4 Segmentor and generates the 3D coordinates needed to
    crop the original standardized MRI for the T3 ROI Classifier.

    Parameters
    ----------
    segmentation_mask : np.ndarray
        3D numpy array of shape (D, H, W). Can be multi-class or binary.
    padding_pct : float, optional
        Percentage of padding to add around the bounding box margins, by default 0.10 (10%).

    Returns
    -------
    tuple
        (z_min, z_max, y_min, y_max, x_min, x_max) coordinates for cropping.
        Returns None if no tumor voxels are found in the mask.
    """
    # Collapse multi-class mask to binary (any label > 0 is tumor)
    binary_mask = (segmentation_mask > 0).astype(np.uint8)
    
    # Find coordinates of non-zero voxels
    tumor_coords = np.where(binary_mask > 0)
    
    if len(tumor_coords[0]) == 0:
        return None  # No tumor found
        
    z_min, z_max = np.min(tumor_coords[0]), np.max(tumor_coords[0])
    y_min, y_max = np.min(tumor_coords[1]), np.max(tumor_coords[1])
    x_min, x_max = np.min(tumor_coords[2]), np.max(tumor_coords[2])
    
    # Calculate padding
    depth, height, width = segmentation_mask.shape
    
    z_pad = int((z_max - z_min) * padding_pct)
    y_pad = int((y_max - y_min) * padding_pct)
    x_pad = int((x_max - x_min) * padding_pct)
    
    # Apply padding with boundary checks
    z_min = max(0, z_min - z_pad)
    z_max = min(depth - 1, z_max + z_pad)
    y_min = max(0, y_min - y_pad)
    y_max = min(height - 1, y_max + y_pad)
    x_min = max(0, x_min - x_pad)
    x_max = min(width - 1, x_max + x_pad)
    
    return (z_min, z_max, y_min, y_max, x_min, x_max)


def crop_volume(volume: np.ndarray, bbox: tuple) -> np.ndarray:
    """Crop a 3D or 4D (C, D, H, W) volume using bounding box coordinates.

    Parameters
    ----------
    volume : np.ndarray
        The original standardized MRI volume.
    bbox : tuple
        Bounding box coordinates (z_min, z_max, y_min, y_max, x_min, x_max).

    Returns
    -------
    np.ndarray
        Cropped volume.
    """
    if bbox is None:
        raise ValueError("Cannot crop volume: No bounding box provided (no tumor detected).")
        
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    
    if volume.ndim == 4:
        # Channels first: (C, D, H, W)
        return volume[:, z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
    elif volume.ndim == 3:
        # Spatial only: (D, H, W)
        return volume[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
    else:
        raise ValueError(f"Unsupported volume dimensions: {volume.ndim}")
