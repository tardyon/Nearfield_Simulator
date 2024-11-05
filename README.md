# Nearfield Beam Simulator
/**
 * Version: 1.00
 * Author: Michael C.M Varney
 * Email: ****

An interactive web application for simulating and visualizing laser beam near-field patterns. This tool allows users to generate realistic beam profiles with various characteristics including asymmetry, noise patterns, and elliptical shapes.

## Features

- **Interactive Interface**: Built with Streamlit for real-time parameter adjustments
- **Customizable Beam Parameters**:
  - Beam size and shape (elliptical parameters)
  - Asymmetry in X and Y directions
  - Perlin noise patterns for realistic beam structure
  - Gaussian noise for detector simulation
  - Edge rolloff control
  - Brightness scaling
- **Multiple Export Options**:
  - JPEG and TIFF format support
  - Configurable bit depth (8, 16, or 32-bit)
- **Real-time Visualization**:
  - 2D beam profile display
  - Cross-sectional intensity plot
  - Customizable color palettes
  - Physical units (millimeters) support

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Nearfield_Simulator
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Access the web interface through your browser (typically http://localhost:8501)

3. Adjust parameters using the sidebar controls:
   - **Image Settings**: Configure resolution and scaling
   - **Elliptical Parameters**: Adjust beam shape and size
   - **Asymmetry Parameters**: Control beam uniformity
   - **Perlin Noise Parameters**: Add realistic beam structure
   - **Noise and Brightness**: Fine-tune intensity and noise levels
   - **Save Options**: Export your generated beam profile

## Parameter Description

### Image Settings
- **Width/Height**: Image dimensions in pixels
- **Pixels per mm**: Physical scale of the simulation
- **Color Palette**: Visualization color scheme

### Elliptical Parameters
- **Radius**: Size of the beam (in mm)
- **Ellipticity**: Aspect ratio of the elliptical shape
- **Rotation**: Angular orientation (degrees)
- **Edge Rolloff**: Smoothness of beam edges

### Asymmetry Parameters
- **X/Y Asymmetry**: Control beam uniformity in both axes

### Perlin Noise Parameters
- **Scale**: Size of noise features
- **Octaves**: Detail levels in the noise
- **Persistence**: How detail intensity changes
- **Lacunarity**: How detail size changes
- **Amplitude**: Overall noise strength
- **Unmodulated %**: Base beam uniformity

### Additional Controls
- **Gaussian Noise**: Simulate detector noise
- **Brightness Scale**: Overall intensity adjustment
- **Export Format**: JPEG or TIFF
- **Bit Depth**: 8, 16, or 32-bit output

## File Structure

- `app.py`: Main application and UI
- `beam_simulation.py`: Core beam simulation logic
- `image_helpers.py`: Image processing utilities
- `beam_params.json`: Default/saved parameters
- `requirements.txt`: Required Python packages

## Dependencies

- numpy: Numerical computations
- scipy: Special functions
- noise: Perlin noise generation
- Pillow: Image processing
- streamlit: Web interface
- matplotlib: Visualization
- tifffile: TIFF file handling

## Parameters Persistence

The application automatically saves parameters between sessions in `beam_params.json`. This allows you to maintain your settings across multiple runs.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
