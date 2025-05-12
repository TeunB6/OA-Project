class ObservationData:
    def __init__(self, path: str):
        self.directory_path = path

        for f in tqdm(files):
            with fits.open(os.path.join(data_dir, f)) as hdul:
                header = hdul[0].header
                row = {"FILENAME" : os.path.basename(f),
                "OBJECT" : header.get("OBJECT", None),
                "FILTER" : header.get("FILTER", None),
                "IMAGETYP" : header.get("IMAGETYP", None),
                "DATE-OBS" : header.get("DATE-OBS", None),
                "EXP-TIME" : header.get("EXP-TIME", None)}
          

