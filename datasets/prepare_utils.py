import os
import shutil
import zipfile
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

def prepare_brain_multi(unzipped_path, output_h5):
    # Prepare the H5 file with raw images and labels for classification.
    categories = ['glioma', 'meningioma', 'notumor', 'pituitary']
    mapping = {cat: i for i, cat in enumerate(categories)}
    with h5py.File(output_h5, 'w') as hdf:
        img_grp = hdf.create_group('images')
        lbl_grp = hdf.create_group('labels')
        
        # group training and testing data together for simplicity
        for folder in ['Training', 'Testing']:
            src = os.path.join(unzipped_path, folder)
            if not os.path.exists(src): continue
            
            for cat in categories:
                cat_path = os.path.join(src, cat)
                if not os.path.exists(cat_path): continue
                
                for img_name in tqdm(os.listdir(cat_path), desc=f"Processing {cat}"):
                    img = Image.open(os.path.join(cat_path, img_name)).convert('L').resize((224, 224))
                    key = f"{folder}_{cat}_{img_name}"
                    img_grp.create_dataset(key, data=np.array(img))
                    lbl_grp.create_dataset(key, data=mapping[cat])
    shutil.rmtree(unzipped_path)

def prepare_brain_detect(unzipped_path, target_dir):
    """Prepare the axial dataset for YOLO detection."""
    axial_zip = os.path.join(unzipped_path, "Axial Dataset.zip")
    temp_axial = os.path.join(unzipped_path, "temp_axial")
    with zipfile.ZipFile(axial_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_axial)
    src_folder = os.path.join(temp_axial, "Axial Data-Augmentation")
    if os.path.exists(target_dir): shutil.rmtree(target_dir)
    shutil.move(src_folder, target_dir)
    shutil.rmtree(unzipped_path)
    print(f"Dataset de détection prêt dans {target_dir}")