import cv2
import numpy as np

def apply_clahe(image_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.
    This fixes night/shadow detection issues by balancing localized lighting.
    
    image_path: Path to the input image file.
    clip_limit: Threshold for contrast limiting.
    tile_grid_size: Size of grid for histogram equalization.
    
    Returns:
        equalized_image: OpenCV BGR image after CLAHE.
    """
    # Read the image in BGR format
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from path: {image_path}")
        
    # Convert image to LAB color space (L = Luminance, A = green/red, B = blue/yellow)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Split the LAB channels
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to the Luminance (L) channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl_l = clahe.apply(l_channel)
    
    # Merge the CLAHE-enhanced L-channel back with A and B channels
    merged_lab = cv2.merge((cl_l, a_channel, b_channel))
    
    # Convert back to BGR color space
    enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr
