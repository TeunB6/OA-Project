import os
import numpy as np
from astropy.io import fits
from scipy.ndimage import center_of_mass
from scipy.stats import mode
from photutils.isophote import EllipseGeometry, Ellipse
from matplotlib import pyplot as plt

from ObservationData import ObservationData


def combine_band_images(obs_data: ObservationData, band: str) -> np.ndarray:
    """
    Loads and stacks images for a specific band, then computes the median-combined image.
    """
    band_files = obs_data.filter(f'FILTER == "{band}" and IMAGETYP == "LIGHT"')
    if band_files.empty:
        raise ValueError(f"No LIGHT images found for band {band}")
        
    stacked = np.dstack([
        fits.getdata(os.path.join(obs_data.directory_path, fname))
        for fname in band_files["FILENAME"]
    ])
    return np.median(stacked, axis=2)

def temp_combine_raw_band_images(raw_dir: str, band: str) -> np.ndarray:
    """
    Combines raw FITS files from a folder, selecting only 'LIGHT' frames with the specified filter.
    """
    stacked = []

    for fname in os.listdir(raw_dir):
        if not fname.lower().endswith(".fits"):
            continue

        path = os.path.join(raw_dir, fname)
        with fits.open(path) as hdul:
            header = hdul[0].header
            data = hdul[0].data

            if header.get("IMAGETYP", "").strip().upper() == "LIGHT" and header.get("FILTER", "").strip().upper() == band.upper():
                stacked.append(data)

    if not stacked:
        raise ValueError(f"No LIGHT frames with filter '{band}' found in {raw_dir}")

    return np.median(np.dstack(stacked), axis=2)


def subtract_sky_background(image: np.ndarray) -> np.ndarray:
    """
    Estimates and subtracts the sky background using the mode.
    """
    sky_mode = mode(image.ravel(), axis=None, keepdims=False).mode
    return image - sky_mode


def estimate_galaxy_center(image: np.ndarray) -> tuple:
    """
    Estimates the galaxy center using brightness-weighted center of mass.
    """
    return center_of_mass(image)


def estimate_radius(image: np.ndarray, center: tuple, threshold_frac: float = 0.2) -> float:
    """
    Estimate galaxy radius as max distance from center to pixels above a brightness threshold.
    
    The threshold is a fraction of the maximum pixel value in the image.
    I found people using 02, since we're looking for the faint outer edge of the galaxy

    if we use a very low fraction eg 005, we'll include noise
    if you use a high fraction eg 05, well only measure the bright core
    so apparentely 20% of the maximum is a common compromise as its faint enough
    to approximate the edge, but not so low as to include background noise

    but we can play with this
    """

    threshold = image.max() * threshold_frac
    y, x = np.where(image > threshold)
    
    if len(x) == 0:
        return 0.0  # fallback

    distances = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    return distances.max()


def fit_ellipses(image: np.ndarray, center: tuple):
    """
    Fit elliptical isophotes to the galaxy image using photutils.
    """
    geometry = EllipseGeometry(x0=center[1], y0=center[0])
    ellipse = Ellipse(image, geometry)
    isophotes = ellipse.fit_image()
    
    return isophotes.to_table()


def plot_surface_brightness(profile_table):
    """
    Plot the surface brightness profile from ellipse fitting results.
    """
    sma = profile_table["sma"]
    intens = profile_table["intens"]
    
    plt.figure(figsize=(8, 5))
    plt.plot(sma, intens, marker='o')
    plt.gca().invert_yaxis()  # Brightness goes down outward
    plt.xlabel("Semi-major Axis (pixels)")
    plt.ylabel("Surface Brightness")
    plt.title("Galaxy Surface Brightness Profile")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
def main():
    raw_path = "./DATASERVER/2025-04-12"
    obs = ObservationData("path/to/fits")

    band = "R"

    image = temp_combine_raw_band_images(raw_path, band)
    clean_image = subtract_sky_background(image)
    center = estimate_galaxy_center(clean_image)
    radius = estimate_radius(clean_image, center)

    print(f"Estimated Center: {center}")
    print(f"Estimated Radius: {radius:.2f} pixels")

    profile_table = fit_ellipses(clean_image, center)
    plot_surface_brightness(profile_table)


if __name__ == "__main__":
    main()
