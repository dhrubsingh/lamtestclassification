import pandas as pd
import os
import shutil
import glob

excel_file = '/root/lam-preprocessing/Labelled/000_Labelled LAMs.xlsx'
source_dir = '/root/lam-preprocessing/Labelled'
dest_dir = '/root/lam-preprocessing/Dataset'

for category in ['Positive', 'Negative', 'Indeterminant']:
    os.makedirs(os.path.join(dest_dir, category), exist_ok=True)

df = pd.read_excel(excel_file)

processed_count = 0
for index, row in df.iterrows():
    try:
        image_id = str(row.iloc[0])
        
        image_pattern = os.path.join(source_dir, f"{image_id}_image.*")
        image_files = glob.glob(image_pattern)
        
        if not image_files:
            print(f"Warning: No image file found for ID {image_id}")
            continue
        
        image_file = image_files[0]
        
        lam_result1 = str(row.iloc[1]).strip().lower() if pd.notna(row.iloc[1]) else None
        lam_result2 = str(row.iloc[2]).strip().lower() if pd.notna(row.iloc[2]) else None
        overruler = str(row.iloc[3]).strip().lower() if pd.notna(row.iloc[3]) and len(row) > 3 else None
        
        if lam_result1 == lam_result2:
            classification = lam_result1
        else:
            classification = overruler
        
        if classification in ['positive']:
            target_dir = os.path.join(dest_dir, 'Positive')
        elif classification in ['negative']:
            target_dir = os.path.join(dest_dir, 'Negative')
        elif classification in ['indeterminant', 'indeterminate']:
            target_dir = os.path.join(dest_dir, 'Indeterminant')
        else:
            print(f"Warning: Unknown classification '{classification}' for image {image_id}")
            continue
        
        target_file = os.path.join(target_dir, os.path.basename(image_file))
        shutil.copy2(image_file, target_file)
        processed_count += 1
        
        if processed_count % 50 == 0:
            print(f"Processed {processed_count} images...")
    
    except Exception as e:
        print(f"Error processing row {index}: {e}")

print(f"Finished processing. Total images processed: {processed_count}")

# Print summary
positive_count = len(os.listdir(os.path.join(dest_dir, 'Positive')))
negative_count = len(os.listdir(os.path.join(dest_dir, 'Negative')))
indeterminant_count = len(os.listdir(os.path.join(dest_dir, 'Indeterminant')))

print(f"Images in Positive directory: {positive_count}")
print(f"Images in Negative directory: {negative_count}")
print(f"Images in Indeterminant directory: {indeterminant_count}")
print(f"Total: {positive_count + negative_count + indeterminant_count}")
