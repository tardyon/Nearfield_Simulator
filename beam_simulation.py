import numpy as np
try:
    from noise import pnoise2
except ImportError:
    raise ImportError("The 'noise' module is required. Please install it using 'pip install noise'.")
from scipy.special import erf

class BeamSimulator:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def generate_perlin_noise(self, scale, octaves, persistence, lacunarity, amplitude=1.0):
        lin_x = np.linspace(0, scale, self.width, endpoint=False)
        lin_y = np.linspace(0, scale, self.height, endpoint=False)
        x, y = np.meshgrid(lin_x, lin_y)
        noise = np.vectorize(lambda x, y: pnoise2(x, y, octaves=octaves, persistence=persistence, lacunarity=lacunarity))(x, y)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        return noise * amplitude

    def apply_asymmetry(self, image, asymmetry_x, asymmetry_y):
        grad_x = np.linspace(1 - asymmetry_x, asymmetry_x, self.width)
        grad_y = np.linspace(1 - asymmetry_y, asymmetry_y, self.height)
        gradient = np.outer(grad_y, grad_x)
        return image * gradient

    def apply_ellipse_mask(self, image, radius, ellipticity, angle, rolloff_width):
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
        noise = np.random.normal(0, std, image.shape)
        return image + noise

    def generate_beam(self, perlin_scale, perlin_octaves, perlin_persistence, perlin_lacunarity, asymmetry_x, asymmetry_y, radius, ellipticity, ellipse_angle, rolloff_width, gaussian_noise_std, enable_perlin_noise=True, brightness_scale=1.0, perlin_amplitude=1.0, unmodulated_percentage=0.1):
        base_image = np.ones((self.height, self.width))
        if enable_perlin_noise:
            perlin_noise = self.generate_perlin_noise(perlin_scale, perlin_octaves, perlin_persistence, perlin_lacunarity, amplitude=perlin_amplitude)
            image = (1 - unmodulated_percentage) * base_image + unmodulated_percentage * perlin_noise
        else:
            image = base_image
        image = self.apply_asymmetry(image, asymmetry_x, asymmetry_y)
        image = self.apply_ellipse_mask(image, radius, ellipticity, ellipse_angle, rolloff_width)
        image = self.add_gaussian_noise(image, gaussian_noise_std)
        image = np.clip(image, 0, 1)
        return image
