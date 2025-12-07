import os
import json
import random
from datasets import load_dataset
from tqdm import tqdm

# ================= CONFIGURATION =================
# Replace this with your actual HuggingFace dataset path (e.g., "organization/dataset_name")
ds_name = "mmiq_1"
HF_DATASET_NAME = f"sghosts/{ds_name}" 

# Directory where images will be saved
OUTPUT_IMAGE_DIR = f"data/{ds_name}/images"

# Name of the final JSON output file
OUTPUT_JSONL_FILE = f"data/{ds_name}/formatted_dataset.jsonl"

# List of random Turkish prompts/instructions
# I have expanded this list to provide variety for training
PROMPTS = [
    "Bu sorunun çözümü ne?",
    "Nasıl çözebilirim?",
    "Çözer misin?",
    "Bu soruya bakabilir misin?",
    "Cevabı nedir?",
    "Bana bu soruyu açıkla.",
    "Bunun cevabını bulabilir misin?",
    "Çözüm adımları nelerdir?",
    "Bu problemi çözmeme yardım et.",
    "Doğru cevap hangisi?",
    "Lütfen bu soruyu yanıtla.",
    "Bunu benim için analiz et ve çöz.",
    "Sorunun cevabını ver.",
    "Bu görseldeki soruyu çöz.",
    "Aşağıdaki soruyu yanıtlayın."
]

# ================= PROCESSING =================

def process_dataset():
    # 1. Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_IMAGE_DIR):
        os.makedirs(OUTPUT_IMAGE_DIR)
        print(f"Created directory: {OUTPUT_IMAGE_DIR}")

    # 2. Load the dataset
    print("Loading dataset...")
    try:
        ds = load_dataset(HF_DATASET_NAME, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Processing {len(ds)} rows and writing to {OUTPUT_JSONL_FILE}...")

    # 3. Open the JSONL file for writing
    with open(OUTPUT_JSONL_FILE, 'w', encoding='utf-8') as f:
        
        # Iterate through the dataset
        for i, row in tqdm(enumerate(ds), total=len(ds)):
            try:
                # --- Image Handling ---
                image_obj = row['image']
                
                # Use data_id if available, otherwise use index
                data_id = row.get('data_id', i) 
                
                image_filename = f"{data_id}.png"
                image_save_path = os.path.join(OUTPUT_IMAGE_DIR, image_filename)
                
                # Save the PIL image
                image_obj.save(image_save_path)
                
                # Get absolute path
                abs_image_path = os.path.abspath(image_save_path)

                # --- Text Handling ---
                random_statement = random.choice(PROMPTS)
                correct_answer = row['answer']

                # --- Structure Formatting ---
                entry = {
                    "id": data_id,
                    "image": abs_image_path,
                    "conversations": [
                        {
                            "from": "human", 
                            "value": f"<image> {random_statement}"
                        },
                        {
                            "from": "gpt", 
                            "value": str(correct_answer)
                        }
                    ]
                }
                
                # --- Write to JSONL ---
                # json.dumps converts the dict to a string, + '\n' makes it a line
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            except Exception as e:
                print(f"Skipping row {i} due to error: {e}")
                continue

    print("Done! Processing complete.")

if __name__ == "__main__":
    process_dataset()
