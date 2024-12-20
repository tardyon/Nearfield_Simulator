"""
Nearfield Beam Simulator - Main Application
Version: 1.01
Author: Michael C.M Varney
Email: ****

Main application script that provides the web interface and user controls for the
Nearfield Beam Simulator. Built with Streamlit, this script handles all user interactions,
parameter management, and visualization of the beam profiles.
"""

import streamlit as st
import io
import numpy as np
import json
import os
import pandas as pd  # For handling dataframes
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError:
    raise ImportError("Please install 'matplotlib' using 'pip install matplotlib'.")
from beam_simulation import BeamSimulator
from image_helpers import save_image

# Filepaths for storing beam parameters and configurations
PARAMS_FILE = "beam_params.json"
MAIN_CONFIG_FILE = "main_config.json"
LOWER_BOUND_FILE = "lower_bound_config.json"
UPPER_BOUND_FILE = "upper_bound_config.json"

# Default beam parameters
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
    'file_format': 'JPEG',
    'perlin_amplitude': 1.0
}

def load_params(filepath):
    """Load beam parameters from a JSON file.

    Parameters:
    - filepath (str): Path to the JSON file.

    Returns:
    - params (dict): Loaded parameters dictionary.
    """
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            params = json.load(f)
            # Ensure all default parameters are present
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
            return params
    else:
        return default_params.copy()

def save_params(filepath, params):
    """Save beam parameters to a JSON file.

    Parameters:
    - filepath (str): Path to save the JSON file.
    - params (dict): Parameters dictionary to save.

    Returns:
    - None
    """
    with open(filepath, 'w') as f:
        json.dump(params, f)

def load_config(filepath, default_params):
    """Load configuration from a JSON file."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            params = json.load(f)
            # Ensure all default parameters are present
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
            return params
    else:
        return default_params.copy()

def save_config(filepath, params):
    """Save configuration to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(params, f)

# Load parameters at startup
params = load_params(PARAMS_FILE)

# Sidebar controls for adjusting parameters
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
    # Parameters for the elliptical shape and rolloff
    params['radius_mm'] = st.slider("Ellipse Radius (mm)", 1.0, min(params['width'], params['height']) / params['pixels_per_mm'], params.get('radius_mm', default_params['radius_mm']))
    params['ellipticity'] = st.slider("Ellipticity", 0.1, 2.0, params['ellipticity'])
    params['ellipse_angle'] = st.slider("Ellipse Rotation (Degrees)", 0, 360, params['ellipse_angle'])
    params['rolloff_width_mm'] = st.slider("Edge Rolloff Width (mm)", 0.1, 10.0, params.get('rolloff_width_mm', default_params['rolloff_width_mm']))

with st.sidebar.expander("Asymmetry Parameters"):
    # Parameters for beam asymmetry
    params['asymmetry_x'] = st.slider("Asymmetry X", 0.0, 1.0, params['asymmetry_x'])
    params['asymmetry_y'] = st.slider("Asymmetry Y", 0.0, 1.0, params['asymmetry_y'])

with st.sidebar.expander("Perlin Noise Parameters"):
    # Parameters for Perlin noise
    params['enable_perlin_noise'] = st.checkbox("Enable Perlin Noise", value=params['enable_perlin_noise'])
    params['perlin_scale'] = st.slider("Perlin Noise Scale", 0.1, 20.0, params['perlin_scale'])
    params['perlin_octaves'] = st.slider("Perlin Noise Octaves", 1, 10, params['perlin_octaves'])
    params['perlin_persistence'] = st.slider("Perlin Noise Persistence", 0.1, 1.0, params['perlin_persistence'])
    params['perlin_lacunarity'] = st.slider("Perlin Noise Lacunarity", 1.0, 4.0, params['perlin_lacunarity'])
    params['perlin_amplitude'] = st.slider("Perlin Noise Amplitude", 0.1, 10.0, params.get('perlin_amplitude', 1.0))
    params['unmodulated_percentage'] = st.slider("Unmodulated Beam Percentage", 0.0, 1.0, params['unmodulated_percentage'])

with st.sidebar.expander("Noise and Brightness"):
    # Parameters for Gaussian noise and brightness scaling
    params['gaussian_noise_std'] = st.slider("Gaussian Noise Std Dev", 0.0, 0.03, params['gaussian_noise_std'], step=0.0001, format="%.4f")
    params['brightness_scale'] = st.slider("Brightness Scale", 0.1, 6.0, params['brightness_scale'])

# Save parameters whenever they are updated
save_params(PARAMS_FILE, params)

# Sidebar for Configuration Management
st.sidebar.title("Configuration Management")

config_option = st.sidebar.selectbox("Save Current Parameters As:", ["Main Configuration", "Lower Bound Configuration", "Upper Bound Configuration"])
if st.sidebar.button("Save Configuration"):
    if config_option == "Main Configuration":
        save_config(MAIN_CONFIG_FILE, params)
        st.sidebar.success("Main configuration saved.")
    elif config_option == "Lower Bound Configuration":
        save_config(LOWER_BOUND_FILE, params)
        st.sidebar.success("Lower bound configuration saved.")
    elif config_option == "Upper Bound Configuration":
        save_config(UPPER_BOUND_FILE, params)
        st.sidebar.success("Upper bound configuration saved.")

# Save sweep parameters to CSV
if st.sidebar.button("Save Sweep Parameters"):
    # Load configurations
    main_config = load_config(MAIN_CONFIG_FILE, default_params)
    lower_bound_config = load_config(LOWER_BOUND_FILE, default_params)
    upper_bound_config = load_config(UPPER_BOUND_FILE, default_params)

    # Prepare data
    sweep_data = []
    for key in main_config.keys():
        if key in ['width', 'height', 'tiff_bit_depth', 'enable_perlin_noise', 'file_format', 'color_palette']:
            continue  # Skip these parameters
        sweep_data.append({
            'parameter': key,
            'lower_bound': lower_bound_config.get(key, default_params[key]),
            'main_value': main_config.get(key, default_params[key]),
            'upper_bound': upper_bound_config.get(key, default_params[key])
        })

    # Save to CSV
    df_sweep = pd.DataFrame(sweep_data)
    df_sweep.to_csv('sweep_parameters.csv', index=False)
    st.sidebar.success("Sweep parameters saved to 'sweep_parameters.csv'.")

with st.sidebar.expander("Save Options"):
    # Options for saving the generated image
    params['tiff_bit_depth'] = st.selectbox("TIFF Bit Depth", options=[8, 16, 32], index=[8, 16, 32].index(params.get('tiff_bit_depth', 16)))
    params['file_format'] = st.selectbox("File Format", ["JPEG", "TIFF"], index=["JPEG", "TIFF"].index(params.get('file_format', "TIFF")))

# Initialize BeamSimulator and generate the beam image
simulator = BeamSimulator(params['width'], params['height'])
beam_image = simulator.generate_beam(
    # Pass all parameters to the beam generation function
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

# Display the beam image and cross-section plots
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

buffer = io.BytesIO()
save_image(beam_image, buffer, format=params['file_format'], bit_depth=params['tiff_bit_depth'])
buffer.seek(0)
st.download_button(
    label="Download Image",
    data=buffer,
    file_name=f"beam_image.{params['file_format'].lower()}",
    mime="image/tiff" if params['file_format'] == 'TIFF' else "image/jpeg"
)

# Batch Generation Section
st.header("Batch Image Generation")

num_images = st.number_input("Number of Images to Generate", min_value=1, value=10)
output_folder = st.text_input("Output Folder", "batch_output")

if st.button("Generate Batch Images"):
    with st.spinner("Generating batch images..."):
        # Load configurations
        main_config = load_config(MAIN_CONFIG_FILE, default_params)
        
        # Load sweep parameters from CSV
        try:
            sweep_df = pd.read_csv('sweep_parameters.csv')
            sweep_params = {
                row['parameter']: {
                    'low': float(row['lower_bound']),
                    'high': float(row['upper_bound'])
                }
                for _, row in sweep_df.iterrows()
            }
        except Exception as e:
            st.error(f"Error loading sweep parameters: {e}")
            st.stop()

        # Generate list of parameter dictionaries
        params_list = []
        for i in range(int(num_images)):
            sample_params = main_config.copy()  # Start with main config as base
            
            # Sample each parameter within its defined range
            for param, bounds in sweep_params.items():
                if param == 'perlin_octaves':
                    sample_params[param] = int(np.random.randint(bounds['low'], bounds['high'] + 1))
                else:
                    sample_params[param] = float(np.random.uniform(bounds['low'], bounds['high']))

            # Handle special parameters
            sample_params['enable_perlin_noise'] = True  # Fixed to True for batch generation
            pixels_per_mm = main_config['pixels_per_mm']  # Fixed scaling factor
            
            # Calculate derived parameters
            sample_params['radius'] = sample_params['radius_mm'] * pixels_per_mm
            sample_params['rolloff_width'] = pixels_per_mm / sample_params['rolloff_width_mm']
            
            params_list.append(sample_params)

        # Initialize simulator
        simulator = BeamSimulator(main_config['width'], main_config['height'])
        df = simulator.generate_batch_beams(
            params_list, 
            output_folder, 
            image_format='TIFF',
            bit_depth=16
        )

        st.success(f"Batch generation completed. Images saved to '{output_folder}'.")
        st.dataframe(df)
