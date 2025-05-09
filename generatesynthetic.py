import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
import random
import cv2

def create_directory_structure(base_path):
    """Create the directory structure for synthetic data"""
    synthetic_path = os.path.join(base_path, 'Synthetic')
    os.makedirs(os.path.join(synthetic_path, 'Positive'), exist_ok=True)
    os.makedirs(os.path.join(synthetic_path, 'Negative'), exist_ok=True)
    return synthetic_path

def generate_base_test(width=224, height=224, background_variation=True):
    """Generate a base lateral flow test image with realistic variations"""
    # Create base image with variable background color
    if background_variation:
        # Create slightly variable background colors
        bg_color = (
            random.randint(235, 255), 
            random.randint(235, 255), 
            random.randint(230, 250)
        )
    else:
        bg_color = (245, 245, 240)
    
    image = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(image)
    
    # Add some texture to the background
    image = add_texture(image, intensity=random.randint(3, 8))
    
    # Draw test strip with variable positioning
    strip_width = int(width * random.uniform(0.65, 0.8))
    strip_height = int(height * random.uniform(0.18, 0.25))
    strip_left = int((width - strip_width) * random.uniform(0.4, 0.6))
    strip_top = int((height - strip_height) * random.uniform(0.4, 0.6))
    
    # Draw the test strip with a slight off-white color
    strip_color = (
        random.randint(225, 240),
        random.randint(225, 240),
        random.randint(220, 235)
    )
    draw.rectangle(
        [(strip_left, strip_top), (strip_left + strip_width, strip_top + strip_height)],
        fill=strip_color
    )
    
    # Add some noise to make it look more realistic
    image = add_noise(image, intensity=random.randint(5, 15))
    
    return image, strip_left, strip_top, strip_width, strip_height

def add_texture(image, intensity=5):
    """Add subtle texture to the image"""
    img_array = np.array(image)
    texture = np.random.randint(0, intensity, img_array.shape).astype(np.float32)
    # Make the texture pattern more coherent
    texture = cv2.GaussianBlur(texture, (5, 5), 0)
    textured_img = np.clip(img_array + texture, 0, 255).astype(np.uint8)
    return Image.fromarray(textured_img)

def add_noise(image, intensity=10):
    """Add random noise to the image"""
    img_array = np.array(image)
    noise = np.random.randint(-intensity, intensity, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def add_control_line(image, strip_left, strip_top, strip_width, strip_height, 
                     intensity=None, thickness=None, position=None, blur_radius=None):
    """Add a control line to the test with realistic variations"""
    draw = ImageDraw.Draw(image)
    
    # Use provided parameters or generate random ones
    if intensity is None:
        # Variable intensity for the control line (usually darker)
        intensity_value = random.randint(70, 150)
        intensity = (intensity_value, intensity_value, intensity_value)
    
    if thickness is None:
        thickness = random.randint(3, 6)
    
    if position is None:
        position = random.uniform(0.2, 0.35)
    
    if blur_radius is None:
        blur_radius = random.uniform(0.3, 1.0)
    
    # Draw the control line
    line_x = strip_left + int(strip_width * position)
    line_top = strip_top
    line_bottom = strip_top + strip_height
    
    for i in range(thickness):
        draw.line([(line_x + i, line_top), (line_x + i, line_bottom)], fill=intensity)
    
    # Blur the line slightly to make it look more realistic
    return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

def add_test_line(image, strip_left, strip_top, strip_width, strip_height, 
                  intensity=None, thickness=None, position=None, blur_radius=None):
    """Add a test line to the test (for positive results) with realistic variations"""
    draw = ImageDraw.Draw(image)
    
    # Use provided parameters or generate random ones
    if intensity is None:
        # Variable intensity for the test line (usually lighter than control)
        intensity_value = random.randint(50, 130)
        intensity = (intensity_value, intensity_value, intensity_value)
    
    if thickness is None:
        thickness = random.randint(2, 5)
    
    if position is None:
        position = random.uniform(0.65, 0.8)
    
    if blur_radius is None:
        blur_radius = random.uniform(0.3, 1.0)
    
    # Draw the test line
    line_x = strip_left + int(strip_width * position)
    line_top = strip_top
    line_bottom = strip_top + strip_height
    
    for i in range(thickness):
        draw.line([(line_x + i, line_top), (line_x + i, line_bottom)], fill=intensity)
    
    # Blur the line slightly to make it look more realistic
    return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

def add_shadow(image, alpha_range=(0.2, 0.6)):
    """Add random shadow to the image"""
    img_array = np.array(image).astype(float)
    
    # Create a random shadow pattern
    shadow = np.ones(img_array.shape[:2])
    
    # Add some random gradient shadows
    num_shadows = random.randint(1, 3)
    for _ in range(num_shadows):
        x1, y1 = random.randint(0, shadow.shape[1]-1), random.randint(0, shadow.shape[0]-1)
        x2, y2 = random.randint(0, shadow.shape[1]-1), random.randint(0, shadow.shape[0]-1)
        
        # Create gradient shadow
        xx, yy = np.meshgrid(np.arange(shadow.shape[1]), np.arange(shadow.shape[0]))
        
        # Distance from line
        dist = np.abs((y2-y1)*xx - (x2-x1)*yy + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)
        
        # Normalize and invert
        shadow_intensity = np.clip(1 - dist / dist.max(), 0, 1) * random.uniform(*alpha_range)
        shadow = shadow * (1 - shadow_intensity)
    
    # Apply shadow to each channel
    for i in range(3):
        img_array[:,:,i] = img_array[:,:,i] * shadow
    
    # Convert back to uint8
    shadowed_img = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(shadowed_img)

def add_blur(image, radius_range=(0.5, 2.5)):
    """Add random blur to the image"""
    radius = random.uniform(*radius_range)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

def add_jpeg_artifacts(image, quality_range=(50, 90)):
    """Add JPEG compression artifacts"""
    quality = random.randint(*quality_range)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

def add_lighting_variation(image, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3)):
    """Add lighting variations to the image"""
    # Random brightness
    brightness_factor = random.uniform(*brightness_range)
    image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    
    # Random contrast
    contrast_factor = random.uniform(*contrast_range)
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)
    
    return image

def add_perspective_distortion(image, distortion_scale=0.15):
    """Add perspective distortion to simulate different camera angles"""
    width, height = image.size
    
    # Define the corners of the image
    corners = [(0, 0), (width, 0), (width, height), (0, height)]
    
    # Randomly perturb the corners
    new_corners = []
    for x, y in corners:
        dx = random.uniform(-distortion_scale * width, distortion_scale * width)
        dy = random.uniform(-distortion_scale * height, distortion_scale * height)
        new_corners.append((x + dx, y + dy))
    
    # Apply perspective transform
    coeffs = find_coeffs(new_corners, corners)
    return image.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

def find_coeffs(pa, pb):
    """Helper function for perspective transform"""
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def generate_negative_test(width=224, height=224, difficulty='normal'):
    """Generate a negative lateral flow test image with variable difficulty"""
    image, strip_left, strip_top, strip_width, strip_height = generate_base_test(width, height)
    
    # Add control line with random intensity
    control_intensity = random.randint(70, 150)
    image = add_control_line(
        image, strip_left, strip_top, strip_width, strip_height, 
        intensity=(control_intensity, control_intensity, control_intensity)
    )
    
    # Apply difficulty-based transformations
    if difficulty == 'easy':
        # Minimal transformations for easy samples
        if random.random() < 0.3:
            image = add_lighting_variation(image, brightness_range=(0.9, 1.1), contrast_range=(0.9, 1.1))
    
    elif difficulty == 'normal':
        # Standard transformations
        if random.random() < 0.5:
            image = add_shadow(image, alpha_range=(0.1, 0.3))
        if random.random() < 0.5:
            image = add_lighting_variation(image, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2))
        if random.random() < 0.3:
            image = add_blur(image, radius_range=(0.3, 1.0))
    
    elif difficulty == 'hard':
        # More aggressive transformations
        if random.random() < 0.7:
            image = add_shadow(image, alpha_range=(0.2, 0.5))
        if random.random() < 0.7:
            image = add_lighting_variation(image, brightness_range=(0.6, 1.4), contrast_range=(0.6, 1.4))
        if random.random() < 0.5:
            image = add_blur(image, radius_range=(0.5, 2.0))
        if random.random() < 0.4:
            image = add_perspective_distortion(image, distortion_scale=0.1)
    
    elif difficulty == 'extreme':
        # Extreme transformations
        if random.random() < 0.8:
            image = add_shadow(image, alpha_range=(0.3, 0.7))
        if random.random() < 0.8:
            image = add_lighting_variation(image, brightness_range=(0.4, 1.6), contrast_range=(0.4, 1.6))
        if random.random() < 0.7:
            image = add_blur(image, radius_range=(0.8, 2.5))
        if random.random() < 0.6:
            image = add_perspective_distortion(image, distortion_scale=0.15)
        # Add a very faint test line that shouldn't be there (false positive challenge)
        if random.random() < 0.3:
            test_intensity = random.randint(10, 40)
            image = add_test_line(
                image, strip_left, strip_top, strip_width, strip_height, 
                intensity=(test_intensity, test_intensity, test_intensity),
                blur_radius=random.uniform(1.0, 2.0)
            )
    
    return image

def generate_positive_test(width=224, height=224, difficulty='normal'):
    """Generate a positive lateral flow test image with variable difficulty"""
    image, strip_left, strip_top, strip_width, strip_height = generate_base_test(width, height)
    
    # Add control line with random intensity
    control_intensity = random.randint(70, 150)
    image = add_control_line(
        image, strip_left, strip_top, strip_width, strip_height, 
        intensity=(control_intensity, control_intensity, control_intensity)
    )
    
    # Add test line with intensity based on difficulty
    if difficulty == 'easy':
        # Clear test line for easy samples
        test_intensity = random.randint(80, 130)
        image = add_test_line(
            image, strip_left, strip_top, strip_width, strip_height, 
            intensity=(test_intensity, test_intensity, test_intensity)
        )
        
        # Minimal transformations
        if random.random() < 0.3:
            image = add_lighting_variation(image, brightness_range=(0.9, 1.1), contrast_range=(0.9, 1.1))
    
    elif difficulty == 'normal':
        # Standard test line
        test_intensity = random.randint(60, 110)
        image = add_test_line(
            image, strip_left, strip_top, strip_width, strip_height, 
            intensity=(test_intensity, test_intensity, test_intensity)
        )
        
        # Standard transformations
        if random.random() < 0.5:
            image = add_shadow(image, alpha_range=(0.1, 0.3))
        if random.random() < 0.5:
            image = add_lighting_variation(image, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2))
        if random.random() < 0.3:
            image = add_blur(image, radius_range=(0.3, 1.0))
    
    elif difficulty == 'hard':
        # Fainter test line
        test_intensity = random.randint(40, 90)
        image = add_test_line(
            image, strip_left, strip_top, strip_width, strip_height, 
            intensity=(test_intensity, test_intensity, test_intensity)
        )
        
        # More aggressive transformations
        if random.random() < 0.7:
            image = add_shadow(image, alpha_range=(0.2, 0.5))
        if random.random() < 0.7:
            image = add_lighting_variation(image, brightness_range=(0.6, 1.4), contrast_range=(0.6, 1.4))
        if random.random() < 0.5:
            image = add_blur(image, radius_range=(0.5, 2.0))
        if random.random() < 0.4:
            image = add_perspective_distortion(image, distortion_scale=0.1)
    
    elif difficulty == 'extreme':
        # Very faint test line
        test_intensity = random.randint(20, 70)
        image = add_test_line(
            image, strip_left, strip_top, strip_width, strip_height, 
            intensity=(test_intensity, test_intensity, test_intensity),
            blur_radius=random.uniform(0.8, 1.5)
        )
        
        # Extreme transformations
        if random.random() < 0.8:
            image = add_shadow(image, alpha_range=(0.3, 0.7))
        if random.random() < 0.8:
            image = add_lighting_variation(image, brightness_range=(0.4, 1.6), contrast_range=(0.4, 1.6))
        if random.random() < 0.7:
            image = add_blur(image, radius_range=(0.8, 2.5))
        if random.random() < 0.6:
            image = add_perspective_distortion(image, distortion_scale=0.15)
    
    return image

def generate_synthetic_dataset(base_path, num_negative=100, num_positive=100):
    """Generate a synthetic dataset of lateral flow tests with varying difficulties"""
    synthetic_path = create_directory_structure(base_path)
    
    # Define difficulty distribution
    difficulties = ['easy', 'normal', 'hard', 'extreme']
    difficulty_weights = [0.2, 0.3, 0.3, 0.2]  # 20% easy, 30% normal, 30% hard, 20% extreme
    
    print(f"Generating {num_negative} negative samples...")
    for i in range(num_negative):
        # Select difficulty level
        difficulty = random.choices(difficulties, weights=difficulty_weights)[0]
        
        # Generate negative test
        image = generate_negative_test(difficulty=difficulty)
        
        # Save image
        image.save(os.path.join(synthetic_path, 'Negative', f'synthetic_neg_{difficulty}_{i:03d}.png'))
        
        if (i+1) % 20 == 0:
            print(f"  Generated {i+1}/{num_negative} negative samples")
    
    print(f"Generating {num_positive} positive samples...")
    for i in range(num_positive):
        # Select difficulty level
        difficulty = random.choices(difficulties, weights=difficulty_weights)[0]
        
        # Generate positive test
        image = generate_positive_test(difficulty=difficulty)
        
        # Save image
        image.save(os.path.join(synthetic_path, 'Positive', f'synthetic_pos_{difficulty}_{i:03d}.png'))
        
        if (i+1) % 20 == 0:
            print(f"  Generated {i+1}/{num_positive} positive samples")
    
    print(f"Synthetic dataset generated at {synthetic_path}")
    return synthetic_path

def visualize_samples_by_difficulty(synthetic_path):
    """Visualize samples from the synthetic dataset grouped by difficulty"""
    difficulties = ['easy', 'normal', 'hard', 'extreme']
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # For each difficulty, show one negative and one positive example
    for i, difficulty in enumerate(difficulties):
        # Find negative samples of this difficulty
        neg_samples = [f for f in os.listdir(os.path.join(synthetic_path, 'Negative')) 
                      if f'_{difficulty}_' in f]
        
        # Find positive samples of this difficulty
        pos_samples = [f for f in os.listdir(os.path.join(synthetic_path, 'Positive')) 
                      if f'_{difficulty}_' in f]
        
        # If samples exist, randomly select one of each
        if neg_samples:
            neg_sample = random.choice(neg_samples)
            neg_img = Image.open(os.path.join(synthetic_path, 'Negative', neg_sample))
            axes[0, i].imshow(neg_img)
            axes[0, i].set_title(f"Negative ({difficulty})")
            axes[0, i].axis('off')
        
        if pos_samples:
            pos_sample = random.choice(pos_samples)
            pos_img = Image.open(os.path.join(synthetic_path, 'Positive', pos_sample))
            axes[1, i].imshow(pos_img)
            axes[1, i].set_title(f"Positive ({difficulty})")
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(synthetic_path, 'difficulty_visualization.png'))
    plt.close()
    print(f"Difficulty visualization saved to {os.path.join(synthetic_path, 'difficulty_visualization.png')}")

if __name__ == "__main__":
    # Import BytesIO for JPEG artifacts function
    from io import BytesIO
    
    # Set the base path to your MajorProject directory
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'MajorProject')
    
    # Generate synthetic dataset with more samples
    synthetic_path = generate_synthetic_dataset(base_path, num_negative=200, num_positive=200)
    
    # Visualize samples by difficulty
    visualize_samples_by_difficulty(synthetic_path)