import streamlit as st
import io
import numpy as np
import json
import os
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError:
    raise ImportError("The 'matplotlib' module is required. Please install it using 'pip install matplotlib'.")
from beam_simulation import BeamSimulator
from image_helpers import save_image

# Filepath for the JSON file
PARAMS_FILE = "beam_params.json"

# Default parameters
default_params = {
    'width': 512,
    'height': 512,
    'perlin_scale': 5.0,
    'perlin_octaves': 6,
    'perlin_persistence': 0.5,
    'perlin_lacunarity': 2.0,
    'asymmetry_x': 0.5,
    'asymmetry_y': 0.5,
    'radius': 200,
    'ellipticity': 1.0,
    'ellipse_angle': 0,
    'rolloff_width': 5.0,
    'gaussian_noise_std': 0.01,
    'tiff_bit_depth': 16,
    'enable_perlin_noise': True,
    'brightness_scale': 5.0,
    'pixels_per_mm': 10.0,
    'unmodulated_percentage': 0.1,
    'color_palette': 'gray',
    'radius_mm': 20.0,
    'rolloff_width_mm': 1.0,
    'file_format': 'JPEG'
}

# Load parameters from JSON file
def load_params(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            params = json.load(f)
            # Ensure all default parameters are present
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
            return params
    else:
        return default_params

# Save parameters to JSON file
def save_params(filepath, params):
    with open(filepath, 'w') as f:
        json.dump(params, f)

# Load parameters
params = load_params(PARAMS_FILE)

# Sidebar controls
st.sidebar.title("Beam Parameters")

with st.sidebar.expander("Image Settings"):
    # Image Dimensions
    params['width'] = st.number_input("Image Sensor Width in Pixels", value=params['width'], min_value=1)
    params['height'] = st.number_input("Image Sensor Height in Pixels", value=params['height'], min_value=1)
    params['pixels_per_mm'] = st.number_input("Pixels per Millimeter", value=params['pixels_per_mm'], min_value=0.1)

    # Color Palette
    color_palettes = ['gray'] + [palette for palette in plt.colormaps() if palette != 'gray']
    params['color_palette'] = st.selectbox("Color Palette", options=color_palettes, index=color_palettes.index(params.get('color_palette', 'gray')))

with st.sidebar.expander("Elliptical Parameters"):
    params['radius_mm'] = st.slider("Ellipse Radius (mm)", 1.0, min(params['width'], params['height']) / params['pixels_per_mm'], params['radius'] / params['pixels_per_mm'])
    params['ellipticity'] = st.slider("Ellipticity", 0.1, 2.0, params['ellipticity'])
    params['ellipse_angle'] = st.slider("Ellipse Rotation (Degrees)", 0, 360, params['ellipse_angle'])
    params['rolloff_width_mm'] = st.slider("Edge Rolloff Width (mm)", 0.1, 10.0, 1 / (params['rolloff_width'] / params['pixels_per_mm']))

with st.sidebar.expander("Asymmetry Parameters"):
    params['asymmetry_x'] = st.slider("Asymmetry X", 0.0, 1.0, params['asymmetry_x'])
    params['asymmetry_y'] = st.slider("Asymmetry Y", 0.0, 1.0, params['asymmetry_y'])

with st.sidebar.expander("Perlin Noise Parameters"):
    params['enable_perlin_noise'] = st.checkbox("Enable Perlin Noise", value=params['enable_perlin_noise'])
    params['perlin_scale'] = st.slider("Perlin Noise Scale", 0.1, 20.0, params['perlin_scale'])
    params['perlin_octaves'] = st.slider("Perlin Noise Octaves", 1, 10, params['perlin_octaves'])
    params['perlin_persistence'] = st.slider("Perlin Noise Persistence", 0.1, 1.0, params['perlin_persistence'])
    params['perlin_lacunarity'] = st.slider("Perlin Noise Lacunarity", 1.0, 4.0, params['perlin_lacunarity'])
    params['perlin_amplitude'] = st.slider("Perlin Noise Amplitude", 0.1, 10.0, params.get('perlin_amplitude', 1.0))
    params['unmodulated_percentage'] = st.slider("Unmodulated Beam Percentage", 0.0, 1.0, params['unmodulated_percentage'])

with st.sidebar.expander("Noise and Brightness"):
    params['gaussian_noise_std'] = st.slider("Gaussian Noise Std Dev", 0.0, 0.03, params['gaussian_noise_std'], step=0.0001, format="%.4f")
    params['brightness_scale'] = st.slider("Brightness Scale", 0.1, 6.0, params['brightness_scale'])

# Save parameters whenever they are updated
save_params(PARAMS_FILE, params)

# Beam simulation
simulator = BeamSimulator(params['width'], params['height'])
beam_image = simulator.generate_beam(
    perlin_scale=params['perlin_scale'],
    perlin_octaves=params['perlin_octaves'],
    perlin_persistence=params['perlin_persistence'],
    perlin_lacunarity=params['perlin_lacunarity'],
    asymmetry_x=params['asymmetry_x'],
    asymmetry_y=params['asymmetry_y'],
    radius=params['radius_mm'] * params['pixels_per_mm'],
    ellipticity=params['ellipticity'],
    ellipse_angle=params['ellipse_angle'],
    rolloff_width=params['pixels_per_mm'] / params['rolloff_width_mm'],  # Inverse of the displayed value
    gaussian_noise_std=params['gaussian_noise_std'],
    enable_perlin_noise=params['enable_perlin_noise'],
    brightness_scale=params['brightness_scale'],
    perlin_amplitude=params['perlin_amplitude'],
    unmodulated_percentage=params['unmodulated_percentage']
)

# Display image with color bar and horizontal line
st.markdown("<h1 style='text-align: center;'>Beam Near Field</h1>", unsafe_allow_html=True)
fig, ax = plt.subplots()
cax = ax.imshow(beam_image, cmap=params['color_palette'], extent=[0, params['width'] / params['pixels_per_mm'], 0, params['height'] / params['pixels_per_mm']])
ax.set_xlabel("Width (mm)")
ax.set_ylabel("Height (mm)")
fig.colorbar(cax)
st.pyplot(fig)

# Display cross-section
cross_section_fig, cross_section_ax = plt.subplots(figsize=(fig.get_size_inches()[0], 2))  # Ensure same width
cross_section_ax.plot(np.linspace(0, params['width'] / params['pixels_per_mm'], params['width']), beam_image[int(params['height'] / 2), :])
cross_section_ax.set_xlabel("Width (mm)")
cross_section_ax.set_ylabel("Intensity")
st.pyplot(cross_section_fig)

with st.sidebar.expander("Save Options"):
    params['tiff_bit_depth'] = st.selectbox("TIFF Bit Depth", options=[8, 16, 32], index=[8, 16, 32].index(params.get('tiff_bit_depth', 16)))
    params['file_format'] = st.selectbox("File Format", ["JPEG", "TIFF"], index=["JPEG", "TIFF"].index(params.get('file_format', "JPEG")))
    buffer = io.BytesIO()
    save_image(beam_image, buffer, format=params['file_format'], bit_depth=params['tiff_bit_depth'])
    buffer.seek(0)
    st.download_button(
        label="Download Image",
        data=buffer,
        file_name=f"beam_image.{params['file_format'].lower()}",
        mime="image/tiff" if params['file_format'] == 'TIFF' else "image/jpeg"
    )
