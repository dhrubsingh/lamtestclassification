import pandas as pd
import os
import shutil
import glob

# Path to the Excel file and directories
excel_file = '/root/lam-preprocessing/Labelled/000_Labelled LAMs.xlsx'
source_dir = '/root/lam-preprocessing/Labelled'
dest_dir = '/root/lam-preprocessing/Dataset'

# Create destination directories if they don't exist
for category in ['Positive', 'Negative', 'Indeterminant']:
    os.makedirs(os.path.join(dest_dir, category), exist_ok=True)

# Read the Excel file
df = pd.read_excel(excel_file)

# Print column names to help with debugging
print("Columns in the Excel file:", df.columns.tolist())

# Process each row in the dataframe
processed_count = 0
for index, row in df.iterrows():
    try:
        # Extract image ID
        image_id = str(row.iloc[0])  # Assuming the first column contains the image ID
        
        # Try to find the corresponding image file
        image_pattern = os.path.join(source_dir, f"{image_id}_image.*")
        image_files = glob.glob(image_pattern)
        
        if not image_files:
            print(f"Warning: No image file found for ID {image_id}")
            continue
        
        image_file = image_files[0]  # Take the first match
        
        # Determine the classification
        # Assuming columns 1 and 2 are the LAM results, and column 3 is the overruler
        lam_result1 = str(row.iloc[1]).strip().lower() if pd.notna(row.iloc[1]) else None
        lam_result2 = str(row.iloc[2]).strip().lower() if pd.notna(row.iloc[2]) else None
        overruler = str(row.iloc[3]).strip().lower() if pd.notna(row.iloc[3]) and len(row) > 3 else None
        
        # Determine the final classification
        if lam_result1 == lam_result2:
            classification = lam_result1
        else:
            classification = overruler
        
        # Map classification to directory
        if classification in ['positive']:
            target_dir = os.path.join(dest_dir, 'Positive')
        elif classification in ['negative']:
            target_dir = os.path.join(dest_dir, 'Negative')
        elif classification in ['indeterminant', 'indeterminate']:
            target_dir = os.path.join(dest_dir, 'Indeterminant')
        else:
            print(f"Warning: Unknown classification '{classification}' for image {image_id}")
            continue
        
        # Copy the image to the appropriate directory
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
