import os
import pandas as pd
from PIL import Image
import numpy as np

def generate_dummy_dataset(output_dir="dummy_dataset", num_samples=100):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    data = []
    
    for i in range(num_samples):
        # Generate a random dummy image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        img_filename = f"dummy_{i:04d}.jpg"
        img_path = os.path.join(images_dir, img_filename)
        img.save(img_path)
        
        # Determine split (80% train, 20% val)
        split = "train" if i < num_samples * 0.8 else "val"
        
        # View Position (AP or PA)
        view = "PA" if i % 2 == 0 else "AP"
        
        # Dummy report
        if i % 3 == 0:
            report = f"Findings: The lungs are clear. Heart size is normal. Sample {i}."
        elif i % 3 == 1:
            report = f"Impression: No acute cardiopulmonary abnormality. Sample {i}."
        else:
            report = f"Findings: Bilateral pulmonary opacities. Impression: Pneumonia. Sample {i}."
            
        data.append({
            "image_path": img_filename,
            "ViewPosition": view,
            "report": report,
            "split": split
        })
        
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, "metadata.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"Generated {num_samples} dummy samples in {output_dir}")
    print(f"Metadata saved to {csv_path}")
    
if __name__ == "__main__":
    generate_dummy_dataset()
