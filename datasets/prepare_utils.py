import os
import shutil
import zipfile
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

def prepare_brain_multi(unzipped_path, output_h5):
    """Prépare un fichier H5 unique avec les données brutes (classification)."""
    categories = ['glioma', 'meningioma', 'notumor', 'pituitary']
    mapping = {cat: i for i, cat in enumerate(categories)}
    
    print(f"🔨 Création du fichier H5 : {output_h5}")
    with h5py.File(output_h5, 'w') as hdf:
        img_grp = hdf.create_group('images')
        lbl_grp = hdf.create_group('labels')
        
        # On regroupe Training et Testing dans le même H5
        for folder in ['Training', 'Testing']:
            src = os.path.join(unzipped_path, folder)
            if not os.path.exists(src): continue
            
            for cat in categories:
                cat_path = os.path.join(src, cat)
                if not os.path.exists(cat_path): continue
                
                for img_name in tqdm(os.listdir(cat_path), desc=f"Processing {cat}"):
                    # Lecture brute, redimensionnement standard
                    img = Image.open(os.path.join(cat_path, img_name)).convert('L').resize((224, 224))
                    
                    key = f"{folder}_{cat}_{img_name}"
                    img_grp.create_dataset(key, data=np.array(img))
                    lbl_grp.create_dataset(key, data=mapping[cat])
    
    # Nettoyage optionnel du dossier raw après création du H5
    shutil.rmtree(unzipped_path)

def prepare_brain_detect(unzipped_path, target_dir):
    """Prépare le dossier Axial pour YOLO (détection)."""
    # Extraction du zip Axial spécifique à l'intérieur de l'archive Zenodo
    axial_zip = os.path.join(unzipped_path, "Axial Dataset.zip")
    temp_axial = os.path.join(unzipped_path, "temp_axial")
    
    with zipfile.ZipFile(axial_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_axial)
    
    # Déplacement du dossier Axial Data-Augmentation vers data/detection
    src_folder = os.path.join(temp_axial, "Axial Data-Augmentation")
    if os.path.exists(target_dir): shutil.rmtree(target_dir)
    shutil.move(src_folder, target_dir)
    
    # Nettoyage complet du dossier de téléchargement
    shutil.rmtree(unzipped_path)
    print(f"Dataset de détection prêt dans {target_dir}")