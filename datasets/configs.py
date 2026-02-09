import os

# Default dirs
default_data_dir = 'data'
default_raw_dir = 'raw'

dataset_config = {
    # 1. Multi-classification (4 classes) - Record Masoud Nickparvar
    'brain_multi': {
        'url': 'https://zenodo.org/records/12735702/files/brain-tumor-mri-dataset.zip?download=1',
        'task': 'classification',
        'filename': 'brain_multi.zip'
    },
    
    # 2.Object Detecion (YOLO) - Record Yazan Al-Smadi
    'brain_detect': {
        'url': 'https://zenodo.org/records/7619446/files/Brain%20Tumor%20Paper%20Dataset.zip?download=1',
        'task': 'detection',
        'filename': 'brain_detect.zip'
    }
}