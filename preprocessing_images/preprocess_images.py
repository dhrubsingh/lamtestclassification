import cv2
import numpy as np
import os
import glob
import pytesseract
from matplotlib import pyplot as plt
from PIL import Image
import concurrent.futures
import logging
import re
import os.path as osp

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='preprocessing.log',
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# Define paths with absolute paths or relative to script location
script_dir = osp.dirname(osp.abspath(__file__))
dataset_dir = osp.join(script_dir, 'Dataset')
output_dir = osp.join(script_dir, 'Preprocessed')

# Check if output_dir is writable
if not os.access(osp.dirname(output_dir), os.W_OK):
    # Fallback to a directory in the user's home directory
    home_dir = os.path.expanduser("~")
    output_dir = osp.join(home_dir, 'lam_preprocessed')
    logging.warning(f"Original output directory not writable. Using {output_dir} instead.")

# Create output directories
for category in ['Positive', 'Negative', 'Indeterminant']:
    os.makedirs(os.path.join(output_dir, category), exist_ok=True)

def detect_test_strip(image):
    """Detect the rectangular test strip in the image using multiple methods"""
    height, width = image.shape[:2]
    
    # Color-based detection for blue/green background
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for blue/green colors (broader range)
    lower_blue1 = np.array([90, 50, 50])
    upper_blue1 = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue1, upper_blue1)
    
    lower_blue2 = np.array([75, 50, 50])
    upper_blue2 = np.array([95, 255, 255])
    mask_cyan = cv2.inRange(hsv, lower_blue2, upper_blue2)
    
    # Combine masks
    mask = cv2.bitwise_or(mask_blue, mask_cyan)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size and shape
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h > 0 else 0
        
        # Test strips are typically rectangular with specific aspect ratios
        if 0.1 < aspect_ratio < 10 and area > 1000:
            valid_contours.append((contour, area, (x, y, w, h)))
    
    # Try edge detection
    if not valid_contours:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Apply dilation to connect edges
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        
        # Find contours in the edge image
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 0
            
            # Test strips are typically rectangular
            if 0.1 < aspect_ratio < 10 and area > 1000:
                valid_contours.append((contour, area, (x, y, w, h)))
    
    # Try adaptive thresholding
    if not valid_contours:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 0
            
            if 0.1 < aspect_ratio < 10 and area > 1000:
                valid_contours.append((contour, area, (x, y, w, h)))
    
    # Look for long rectangular shapes
    if not valid_contours:
        # Try to find long rectangular objects
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 0
            
            # Test strips are typically rectangular
            if 0.1 < aspect_ratio < 10:
                valid_contours.append((contour, area, (x, y, w, h)))
    
    # If we have valid contours, choose the best one
    if valid_contours:
        # Sort by area (largest first)
        valid_contours.sort(key=lambda x: x[1], reverse=True)
        
        # Get the bounding rectangle of the largest contour
        _, _, (x, y, w, h) = valid_contours[0]
        
        # Add a margin around the strip
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(width - x, w + 2*margin)
        h = min(height - y, h + 2*margin)
        
        # Crop the image
        cropped = image[y:y+h, x:x+w]
        
        return cropped, (x, y, w, h)
    
    # Use a default approach - look for the center portion of the image
    center_x = width // 2
    center_y = height // 2
    
    # Define a region around the center
    crop_width = min(width, 300)
    crop_height = min(height, 100)
    
    x = max(0, center_x - crop_width // 2)
    y = max(0, center_y - crop_height // 2)
    w = min(width - x, crop_width)
    h = min(height - y, crop_height)
    
    # Crop the image
    cropped = image[y:y+h, x:x+w]
    
    return cropped, (x, y, w, h)

def detect_text_orientation(image, max_attempts=2):
    """Detect text orientation using OCR to find 'PATIENT' and 'CONTROL'"""
    # First try a simple heuristic based on image dimensions
    h, w = image.shape[:2]
    dimension_orientation = 90 if h > w else 0
    
    # If we're not going to attempt OCR, just return the dimension-based orientation
    if max_attempts <= 0:
        return dimension_orientation
    
    # For efficiency, resize large images before OCR
    max_dim = 1000
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        resized = image
    
    # Convert to grayscale for OCR
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    
    # Apply minimal preprocessing to enhance text
    # Use CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Define orientations
    orientations = [dimension_orientation, 0, 90, 180, 270]
    orientations = list(dict.fromkeys(orientations))
    
    # Keywords to look for
    keywords = ['PATIENT', 'CONTROL', 'TEST', 'RESULT']
    
    best_orientation = dimension_orientation
    max_score = 0
    
    # Try each orientation
    for angle in orientations:
        if angle == 0:
            rotated = enhanced
        else:
            # Rotate image
            rotated = Image.fromarray(enhanced)
            rotated = rotated.rotate(angle)
            rotated = np.array(rotated)
        
        # Use pytesseract with a simple configuration for speed
        config = '--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        try:
            # Set a timeout for pytesseract to avoid hanging
            text = pytesseract.image_to_string(rotated, config=config, timeout=2).upper()
        except Exception as e:
            logging.warning(f"OCR error: {str(e)}, using fallback orientation")
            text = ""
        
        # Calculate score based on presence of keywords
        score = 0
        for keyword in keywords:
            if keyword in text:
                score += 1
        
        if score > max_score:
            max_score = score
            best_orientation = angle
            
        if score >= 2:
            break
    
    if max_score == 0 and max_attempts > 1:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return detect_text_orientation(binary, max_attempts - 1)
    
    return best_orientation

def enhance_image(image):
    """Enhance the image to make test and control lines more visible"""
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Increase contrast
    alpha = 1.3
    beta = 10
    enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Invert the image to make lines more visible (dark lines on light background)
    inverted = cv2.bitwise_not(thresh)
    
    # Apply morphological operations to enhance the lines
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel)
    
    return processed

def preprocess_image(image_path):
    """Preprocess a single image without cropping, only enhancing colors"""
    try:
        # Extract category and filename
        category = os.path.basename(os.path.dirname(image_path))
        filename = os.path.basename(image_path)
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to read image: {image_path}")
            return False
        
        # Save original dimensions for logging
        original_height, original_width = image.shape[:2]
        
        # Detect orientation using text detection on the original image
        try:
            orientation = detect_text_orientation(image)
        except Exception as e:
            logging.warning(f"Text orientation detection failed for {filename}: {str(e)}. Using default orientation.")
            orientation = 90 if original_height > original_width else 0
        
        # Rotate if needed - do this BEFORE color enhancement
        if orientation > 0:
            # Use PIL for rotation to avoid artifacts
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            rotated_pil = pil_img.rotate(orientation, expand=True)
            processed_image = cv2.cvtColor(np.array(rotated_pil), cv2.COLOR_RGB2BGR)
        else:
            processed_image = image
        
        # Now apply color enhancement to the properly oriented image
        enhanced = enhance_image(processed_image)
        
        # Save the preprocessed image
        output_path = os.path.join(output_dir, category, filename)
        cv2.imwrite(output_path, enhanced)
        
        logging.info(f"Successfully preprocessed {filename} - Original: {original_width}x{original_height}, Final: {enhanced.shape[1]}x{enhanced.shape[0]}, Orientation: {orientation}°")
        return True
    
    except Exception as e:
        logging.error(f"Error processing {image_path}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def main():
    """Main function to process all images"""
    # Get all image files
    image_files = []
    for category in ['Positive', 'Negative', 'Indeterminant']:
        pattern = os.path.join(dataset_dir, category, '*.*')
        category_files = glob.glob(pattern)
        image_files.extend(category_files)
    
    logging.info(f"Found {len(image_files)} images to process")
    
    # Process images sequentially to avoid memory issues
    success_count = 0
    total_count = len(image_files)
    
    # Process a small batch first to check for issues
    for i, image_path in enumerate(image_files[:10]):
        success = preprocess_image(image_path)
        if success:
            success_count += 1
        if (i + 1) % 5 == 0:
            logging.info(f"Progress: {i+1}/{total_count} images processed")
    
    # Process the rest of the images
    for i, image_path in enumerate(image_files[10:], 10):
        success = preprocess_image(image_path)
        if success:
            success_count += 1
        if (i + 1) % 20 == 0:
            logging.info(f"Progress: {i+1}/{total_count} images processed")
    
    logging.info(f"Preprocessing completed. Successfully processed {success_count}/{total_count} images")
    
    # Print summary of categories
    for category in ['Positive', 'Negative', 'Indeterminant']:
        category_count = len(glob.glob(os.path.join(output_dir, category, '*.*')))
        logging.info(f"Images in {category} directory: {category_count}")

if __name__ == "__main__":
    main()
