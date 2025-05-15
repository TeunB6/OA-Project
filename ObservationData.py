import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

class ObservationData:
    """
    Class for managing a directory containing observational data. Wraps a pd dataframe storing all the data's relevant headers.
    """    
    
    def __init__(self, path: str):    
        self.directory_path = path
        self.refresh_dir()
    
    def refresh_dir(self):
        directory = []
        for f in os.listdir(self.directory_path):
            with fits.open(os.path.join(self.directory_path, f)) as hdul:
                header = hdul[0].header

                row = {"FILENAME" : os.path.basename(f),
                "OBJECT" : header.get("OBJECT", None),
                "FILTER" : header.get("FILTER", None),
                "IMAGETYP" : header.get("IMAGETYP", None),
                "DATE-OBS" : header.get("DATE-OBS", None),
                "EXP-TIME" : header.get("EXP-TIME", None),
                "JD" : header.get("JD", None)}
                directory.append(row)
        
        self.directory_frame = pd.DataFrame(directory)
    
    def plot_gray_scale(data: np.ndarray | str, title: str) -> None:
        if isinstance(data, str):
            data = fits.getdata(data)
            
        fig, ax = plt.subplots(figsize=(10,8))
        img_display = ax.imshow(data, cmap='gray', vmin=np.percentile(data, 5), vmax=np.percentile(data, 95))
        cbar = fig.colorbar(img_display, ax=ax)

        ax.set_title(title, pad=20)
        ax.grid(False)
        plt.tight_layout()
        plt.show()
        
    def filter(self, expression: str, replace: bool) -> pd.DataFrame | None:
        if replace:
            self.directory_frame = self.directory_frame.query(expression)
        else:
            return self.directory_frame.query(expression)

    def save_csv(self, file_name: str) -> None:
        self.directory_frame.to_csv(file_name)
    
    def load_csv(self, file_name: str) -> None:
        self.directory_frame = pd.read_csv(file_name)
