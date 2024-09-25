import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from src.feature_extraction import features_extractor

def process_file(row):
    file_name = f"sample{row['fold']}/{row['filename']}"
    final_class_labels = row['name']
    
    features = features_extractor(file_name)
    
    if features is not None:
        return [[feature, final_class_labels] for feature in features]
    return []

def process_metadata(metadata, max_workers=3):
    extracted_features = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_file, [row for _, row in metadata.iterrows()]),
                            total=metadata.shape[0], desc="Processing Files"))

    extracted_features = [item for sublist in results for item in sublist if sublist]

    return extracted_features
