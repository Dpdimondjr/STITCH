from astroquery.mast import Observations, Catalogs
import pandas as pd

# Query all TESS SPOC observations for a specific sector/camera/CCD
obs = Observations.query_criteria(
    obs_collection="TESS",
    sequence_number=86,  # sector number
    provenance_name="SPOC"
)

# Get just the metadata, no file downloads
print(obs.colnames)
print(obs[:5])