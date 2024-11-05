import numpy as np
from PIL import Image
import tifffile

def scale_image(image, scale=0.7):
    return image * scale
 
def save_image(image, buffer, format='JPEG', bit_depth=16):
    if format == 'TIFF':
        image_uint = (image * (2 ** bit_depth - 1)).astype(f'uint{bit_depth}')
        tifffile.imwrite(buffer, image_uint, photometric='minisblack')
    else:
        image_uint = (image * 255).astype('uint8')
        pil_image = Image.fromarray(image_uint)
        pil_image.convert("L").save(buffer, format=format)
