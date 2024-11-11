"""
Nearfield Beam Simulator - Beam Simulation Engine
Version: 1.01
Author: Michael C.M Varney
Email: ****

Core simulation engine that handles the generation of beam profiles, including
Perlin noise patterns, asymmetry, elliptical masks, and various beam characteristics.
This version includes batch image generation functionality.
"""

import numpy as np
from tqdm import tqdm  # For progress bar during batch generation
import pandas as pd  # For saving parameters to CSV
import os

try:
    from noise import pnoise2
except ImportError:
    raise ImportError("The 'noise' module is required. Please install it using 'pip install noise'.")
from scipy.special import erf

class BeamSimulator:
    def __init__(self, width, height):
        """Initialize the BeamSimulator with the given dimensions.

        Parameters:
        - width (int): Width of the beam image in pixels.
        - height (int): Height of the beam image in pixels.
        """
        self.width = width
        self.height = height

    def generate_perlin_noise(self, scale, octaves, persistence, lacunarity, amplitude=1.0):
        """Generate a Perlin noise pattern.

        Parameters:
        - scale (float): Controls the scale of the noise pattern.
        - octaves (int): Number of layers of noise to combine.
        - persistence (float): Amplitude of each octave relative to the previous one.
        - lacunarity (float): Frequency of each octave relative to the previous one.
        - amplitude (float): Overall amplitude of the noise pattern.

        Returns:
        - noise (ndarray): Generated Perlin noise as a 2D array.
        """
        lin_x = np.linspace(0, scale, self.width, endpoint=False)
        lin_y = np.linspace(0, scale, self.height, endpoint=False)
        x, y = np.meshgrid(lin_x, lin_y)
        noise = np.vectorize(lambda x, y: pnoise2(x, y, octaves=octaves, persistence=persistence, lacunarity=lacunarity))(x, y)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        return noise * amplitude

    def apply_asymmetry(self, image, asymmetry_x, asymmetry_y):
        """Apply asymmetry gradients to the beam image.

        Parameters:
        - image (ndarray): Input image to apply asymmetry to.
        - asymmetry_x (float): Asymmetry factor along the x-axis (0 to 1).
        - asymmetry_y (float): Asymmetry factor along the y-axis (0 to 1).

        Returns:
        - image (ndarray): Image after applying asymmetry.
        """
        grad_x = np.linspace(1 - asymmetry_x, asymmetry_x, self.width)
        grad_y = np.linspace(1 - asymmetry_y, asymmetry_y, self.height)
        gradient = np.outer(grad_y, grad_x)
        return image * gradient

    def apply_ellipse_mask(self, image, radius, ellipticity, angle, rolloff_width):
        """Apply an elliptical mask with edge rolloff to the beam image.

        Parameters:
        - image (ndarray): Input image to apply the mask to.
        - radius (float): Radius of the ellipse.
        - ellipticity (float): Ratio of the minor axis to the major axis (1 is a circle).
        - angle (float): Rotation angle of the ellipse in degrees.
        - rolloff_width (float): Width of the edge rolloff (controls sharpness).

        Returns:
        - image (ndarray): Image after applying the elliptical mask.
        """
        y, x = np.indices((self.height, self.width))
        cx, cy = self.width / 2, self.height / 2
        angle = np.deg2rad(angle)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        x_rot = (x - cx) * cos_angle + (y - cy) * sin_angle
        y_rot = -(x - cx) * sin_angle + (y - cy) * cos_angle
        axes = (radius, radius * ellipticity)
        ellipse_eq = (x_rot / axes[0]) ** 2 + (y_rot / axes[1]) ** 2
        rolloff = 0.5 * (1 + erf((1 - ellipse_eq) * rolloff_width))
        rolloff = np.clip(rolloff, 0, 1)
        return image * rolloff

    def add_gaussian_noise(self, image, std):
        """Add Gaussian noise to the image.

        Parameters:
        - image (ndarray): Input image to add noise to.
        - std (float): Standard deviation of the Gaussian noise.

        Returns:
        - image (ndarray): Image with added Gaussian noise.
        """
        noise = np.random.normal(0, std, image.shape)
        return image + noise

    def generate_beam(self, perlin_scale, perlin_octaves, perlin_persistence, perlin_lacunarity,
                      asymmetry_x, asymmetry_y, radius, ellipticity, ellipse_angle, rolloff_width,
                      gaussian_noise_std, enable_perlin_noise=True, brightness_scale=1.0,
                      perlin_amplitude=1.0, unmodulated_percentage=0.1):
        """Generate the simulated beam image with all parameters applied.

        Parameters:
        - perlin_scale (float): Scale of the Perlin noise.
        - perlin_octaves (int): Number of octaves in Perlin noise.
        - perlin_persistence (float): Persistence of Perlin noise.
        - perlin_lacunarity (float): Lacunarity of Perlin noise.
        - asymmetry_x (float): Asymmetry factor along x-axis.
        - asymmetry_y (float): Asymmetry factor along y-axis.
        - radius (float): Radius of the beam ellipse.
        - ellipticity (float): Ellipticity of the beam ellipse.
        - ellipse_angle (float): Rotation angle of the ellipse.
        - rolloff_width (float): Inverse width of the edge rolloff.
        - gaussian_noise_std (float): Standard deviation of Gaussian noise.
        - enable_perlin_noise (bool): Flag to enable Perlin noise.
        - brightness_scale (float): Scaling factor for brightness.
        - perlin_amplitude (float): Amplitude of Perlin noise.
        - unmodulated_percentage (float): Percentage of unmodulated beam.

        Returns:
        - image (ndarray): Final simulated beam image.
        """
        base_image = np.ones((self.height, self.width))
        if enable_perlin_noise:
            perlin_noise = self.generate_perlin_noise(perlin_scale, perlin_octaves, perlin_persistence, perlin_lacunarity, amplitude=perlin_amplitude)
            image = (1 - unmodulated_percentage) * base_image + unmodulated_percentage * perlin_noise
        else:
            image = base_image
        image = self.apply_asymmetry(image, asymmetry_x, asymmetry_y)
        image = self.apply_ellipse_mask(image, radius, ellipticity, ellipse_angle, rolloff_width)
        image = self.add_gaussian_noise(image, gaussian_noise_std)
        image = np.clip(image * brightness_scale, 0, 1)
        return image

    def generate_batch_beams(self, params_list, output_folder, image_format='TIFF', bit_depth=16):
        """
        Generate a batch of beam images based on a list of parameter dictionaries.

        Parameters:
        - params_list (list): List of parameter dictionaries for each image.
        - output_folder (str): Folder path to save the generated images.
        - image_format (str): Format to save the images ('JPEG' or 'TIFF').
        - bit_depth (int): Bit depth for saving images.

        Returns:
        - df (DataFrame): Pandas DataFrame containing parameters for each image.
        """
        from image_helpers import save_image

        os.makedirs(output_folder, exist_ok=True)
        records = []

        for idx, params in enumerate(tqdm(params_list, desc="Generating Images")):
            # Define a list of parameters to exclude
            excluded_params = [
                'width', 'height', 'tiff_bit_depth',
                'pixels_per_mm', 'color_palette',
                'radius_mm', 'rolloff_width_mm',
                'file_format'  # Newly excluded parameter
            ]
            for param in excluded_params:
                params.pop(param, None)

            # Generate beam image
            image = self.generate_beam(**params)

            # Save image
            filename = f"beam_image_{idx}.{image_format.lower()}"
            filepath = os.path.join(output_folder, filename)
            save_image(image, filepath, format=image_format, bit_depth=bit_depth)

            # Record parameters with filename
            record = params.copy()
            record['filename'] = filename
            records.append(record)

        # Create DataFrame
        df = pd.DataFrame(records)

        # Save parameters to CSV
        csv_filepath = os.path.join(output_folder, 'batch_parameters.csv')
        df.to_csv(csv_filepath, index=False)

        return df
