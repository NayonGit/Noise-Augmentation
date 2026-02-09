#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

# --- FIX DES IMPORTS ---
# On définit la racine du projet et on l'ajoute au début du path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# On n'importe prepare_dataset QU'APRÈS avoir fixé le sys.path
from datasets.prepare import prepare_dataset

def main():
    # Met à jour avec tes nouveaux noms de datasets
    datasets_to_install = ['brain_multi', 'brain_detect']

    # Tes dossiers cibles
    DOWNLOAD_TEMP_DIR = os.path.join(root_dir, 'datasets', 'raw')
    FINAL_DATA_DIR = os.path.join(root_dir, 'datasets', 'data')

    os.makedirs(DOWNLOAD_TEMP_DIR, exist_ok=True)
    os.makedirs(FINAL_DATA_DIR, exist_ok=True)

    print(f"--- Brain Tumor Dataset Preparation ---")
    
    for name in datasets_to_install:
        print(f"\n=== Preparing: {name} ===")
        try:
            prepare_dataset(
                dataset_name=name,
                should_download=True, 
                download_dir=DOWNLOAD_TEMP_DIR,
                data_dir=FINAL_DATA_DIR
            )
            print(f"✅ Success: {name} is ready.")
        except Exception as e:
            print(f"❌ Error during {name}: {e}")

if __name__ == "__main__":
    main()