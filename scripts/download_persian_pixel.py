import os
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

DATASET_NAME = "Omarrran/Persian_Pixel"
SUBSET_NAME = "page"
SAVE_DIR = "./data/persian_pixel"
NUM_WORKERS = 8

def save_sample(item):
    try:
        idx = item['id'] if 'id' in item else item.get('index', 'unknown')
        
        image = item['image']
        image_path = os.path.join(SAVE_DIR, "images", f"page_{idx}.png")
        
        if isinstance(image, Image.Image):
            image.save(image_path)
            
        return f"page_{idx}.png", item.get('text', '')
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None

def download_page_subset():
    print(f"Downloading metadata for subset '{SUBSET_NAME}'...")
    
    dataset = load_dataset(
        DATASET_NAME, 
        name=SUBSET_NAME, 
        split="train"
    )
    
    print(f"Total samples fetched for page subset: {len(dataset)}")

    os.makedirs(os.path.join(SAVE_DIR, "images"), exist_ok=True)
    
    print("Starting parallel download and saving images...")
    
    metadata = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(save_sample, dataset))
        
    for res in results:
        if res:
            metadata.append(f"{res[0]}\t{res[1]}")

    label_file = os.path.join(SAVE_DIR, "labels.tsv")
    with open(label_file, "w", encoding="utf-8") as f:
        f.write("image_path\ttext\n")
        f.write("\n".join(metadata))
        
    print(f"Download complete! Files saved to '{SAVE_DIR}'.")

if __name__ == "__main__":
    download_page_subset()