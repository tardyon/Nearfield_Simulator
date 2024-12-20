"""
Nearfield Beam Simulator - Image Processing Utilities
Version: 1.00
Author: Michael C.M Varney
Email: ****

Utility functions for image processing, including scaling, saving, and format
conversion for the beam profile images generated by the simulator.
"""

import numpy as np
from PIL import Image
import tifffile

def scale_image(image, scale=0.7):
    """Scale the image intensity.

    Parameters:
    - image (ndarray): Input image to scale.
    - scale (float): Scaling factor for the image intensity.

    Returns:
    - image (ndarray): Scaled image.
    """
    return image * scale

def save_image(image, buffer, format='JPEG', bit_depth=16):
    """Save the image to a buffer in the specified format and bit depth.

    Parameters:
    - image (ndarray): Image to save.
    - buffer (BytesIO): Buffer to save the image into.
    - format (str): Format to save the image ('JPEG' or 'TIFF').
    - bit_depth (int): Bit depth for the saved image (8, 16, or 32).

    Returns:
    - None
    """
    if format == 'TIFF':
        # Convert image to the specified bit depth
        image_uint = (image * (2 ** bit_depth - 1)).astype(f'uint{bit_depth}')
        tifffile.imwrite(buffer, image_uint, photometric='minisblack')
    else:
        # Convert image to 8-bit and save as JPEG
        image_uint = (image * 255).astype('uint8')
        pil_image = Image.fromarray(image_uint)
        pil_image.convert("L").save(buffer, format=format)
