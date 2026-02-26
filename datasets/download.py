import requests
import zipfile
import os
from tqdm import tqdm

def download_dataset(url: str, fname: str):
    """Download a dataset from a given URL with progress bar and error handling."""
    if os.path.exists(fname):
        print(f"The file {fname} already exists. Skip.")
        return

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status() # Check if the link is dead (404)
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(fname, 'wb') as file, tqdm(
            desc=f"Downloading {os.path.basename(fname)}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=8192):
                size = file.write(data)
                bar.update(size)
    except Exception as e:
        print(f"Error during download: {e}")
        print("Check your internet connection or the validity of the Zenodo link.")

def unzip_dataset(zip_path, extract_to):
    """Exctract and clean the dataset folder."""
    if not os.path.exists(zip_path):
        return
    print(f"Extraction to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)