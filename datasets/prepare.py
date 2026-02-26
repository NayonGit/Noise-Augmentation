from .configs import default_data_dir, dataset_config
from datasets import prepare_utils
from datasets.download import download_dataset, unzip_dataset
import os
import sys

def prepare_dataset(dataset_name: str,
                    should_download: bool = True,
                    download_dir: str = 'raw',
                    data_dir: str = default_data_dir) -> None:

    assert dataset_name in dataset_config, f'Dataset {dataset_name} non trouvé.'

    config = dataset_config[dataset_name]
    download_url = config['url']
    
    # Downloading paths
    download_path = os.path.join(download_dir, f"{dataset_name}.zip")
    unzipped_path = os.path.join(download_dir, dataset_name)

    # Download and unzip
    if should_download and not os.path.exists(unzipped_path):
        download_dataset(download_url, download_path)
        unzip_dataset(download_path, unzipped_path)

    # Search prepare_brain_multi or prepare_brain_detect
    prepare_func = getattr(prepare_utils, f'prepare_{dataset_name}')
    
    if config['task'] == 'classification':
        output_path = os.path.join(data_dir, f'{dataset_name}.h5')
        os.makedirs(data_dir, exist_ok=True)
        prepare_func(unzipped_path, output_path)
    else:
        output_path = os.path.join(data_dir, 'detection')
        prepare_func(unzipped_path, output_path)
if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else 'brain_multi'
    prepare_dataset(dataset_name)