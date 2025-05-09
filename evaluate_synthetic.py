import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set fixed random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Data loading settings
WIDTH = 224
HEIGHT = 224
CHANNEL = 3

# Load models
models = {
    'Model_A': tf.keras.models.load_model('my_model.h5'),
    'Model_B': tf.keras.models.load_model('my_model-SGD.h5'),
    'Model_C': tf.keras.models.load_model('model-nasnet.h5'),
    'Model_D': tf.keras.models.load_model('densenet-model.h5')
}

def load_data_from_directory(data_path):
    """Load and prepare dataset from a directory"""
    # Get file lists
    neg_images = os.listdir(os.path.join(data_path, 'Negative'))
    pos_images = os.listdir(os.path.join(data_path, 'Positive'))
    
    # Create dataframes
    neg_df = pd.DataFrame({'id': neg_images, 'label': 0})
    pos_df = pd.DataFrame({'id': pos_images, 'label': 1})
    
    data = []
    labels = []
    
    # Load negative images
    for img_id in neg_df['id']:
        img_path = os.path.join(data_path, 'Negative', img_id)
        try:
            img = Image.open(img_path)
            img = np.array(img.resize((WIDTH, HEIGHT))) / 255.0
            data.append(img)
            labels.append(0)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    # Load positive images
    for img_id in pos_df['id']:
        img_path = os.path.join(data_path, 'Positive', img_id)
        try:
            img = Image.open(img_path)
            img = np.array(img.resize((WIDTH, HEIGHT))) / 255.0
            data.append(img)
            labels.append(1)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    data = np.array(data)
    labels = np.array(labels)
    
    print(f"Loaded {len(data)} images from {data_path}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    return data, labels

def evaluate_model(model, X, y, model_name, dataset_name):
    """Evaluate a model and return metrics"""
    # Get predictions
    pred = model.predict(X, verbose=0)
    y_pred = np.argmax(pred, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,  # Same as sensitivity
        'specificity': specificity,
        'f1_score': f1,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, model_name, dataset_name):
    """Plot and save confusion matrix"""
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Negative', 'Positive']
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(
        ax=ax,
        cmap='Blues',
        values_format='d',
        colorbar=True,
    )
    
    plt.title(f'{model_name} - {dataset_name}')
    
    # Create directory if it doesn't exist
    os.makedirs('confusion_matrices', exist_ok=True)
    
    # Save the figure
    plt.savefig(f'confusion_matrices/{model_name}_{dataset_name.replace(" ", "_")}.png')
    plt.close()

def compare_performance(original_results, synthetic_results):
    """Compare and visualize performance between original and synthetic datasets"""
    metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score']
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting
        model_names = list(original_results.keys())
        original_values = [original_results[model][metric] for model in model_names]
        synthetic_values = [synthetic_results[model][metric] for model in model_names]
        
        # Set up bar positions
        x = np.arange(len(model_names))
        width = 0.35
        
        # Create bars
        plt.bar(x - width/2, original_values, width, label='Original Data')
        plt.bar(x + width/2, synthetic_values, width, label='Synthetic Data')
        
        # Add labels and title
        plt.xlabel('Models')
        plt.ylabel(metric.capitalize())
        plt.title(f'Comparison of {metric.capitalize()} between Original and Synthetic Data')
        plt.xticks(x, model_names)
        plt.legend()
        
        # Add value labels on bars
        for i, v in enumerate(original_values):
            plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
        
        for i, v in enumerate(synthetic_values):
            plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.ylim(0, 1.1)  # Set y-axis limit
        
        # Save the figure
        os.makedirs('comparison_plots', exist_ok=True)
        plt.savefig(f'comparison_plots/{metric}_comparison.png')
        plt.close()
    
    # Create a summary table
    summary_data = []
    for model_name in model_names:
        for dataset, results in [('Original', original_results), ('Synthetic', synthetic_results)]:
            row = {
                'Model': model_name,
                'Dataset': dataset,
                'Accuracy': results[model_name]['accuracy'],
                'Precision': results[model_name]['precision'],
                'Recall': results[model_name]['recall'],
                'Specificity': results[model_name]['specificity'],
                'F1 Score': results[model_name]['f1_score']
            }
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary to CSV
    summary_df.to_csv('model_performance_comparison.csv', index=False)
    print("Summary saved to model_performance_comparison.csv")
    
    return summary_df

def analyze_difficulty_levels(synthetic_path, models):
    """Analyze model performance across different difficulty levels"""
    difficulties = ['easy', 'normal', 'hard', 'extreme']
    
    # Create dictionaries to store results
    difficulty_results = {model_name: {diff: {} for diff in difficulties} for model_name in models.keys()}
    
    for difficulty in difficulties:
        # Load positive samples for this difficulty
        pos_files = [f for f in os.listdir(os.path.join(synthetic_path, 'Positive')) 
                    if f'_{difficulty}_' in f]
        
        # Load negative samples for this difficulty
        neg_files = [f for f in os.listdir(os.path.join(synthetic_path, 'Negative')) 
                    if f'_{difficulty}_' in f]
        
        # Load images
        data = []
        labels = []
        
        # Load negative images
        for img_id in neg_files:
            img_path = os.path.join(synthetic_path, 'Negative', img_id)
            try:
                img = Image.open(img_path)
                img = np.array(img.resize((WIDTH, HEIGHT))) / 255.0
                data.append(img)
                labels.append(0)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        # Load positive images
        for img_id in pos_files:
            img_path = os.path.join(synthetic_path, 'Positive', img_id)
            try:
                img = Image.open(img_path)
                img = np.array(img.resize((WIDTH, HEIGHT))) / 255.0
                data.append(img)
                labels.append(1)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        if len(data) == 0:
            print(f"No images found for difficulty level: {difficulty}")
            continue
            
        data = np.array(data)
        labels = np.array(labels)
        
        print(f"Difficulty {difficulty}: Loaded {len(data)} images")
        print(f"Class distribution: {np.bincount(labels)}")
        
        # Evaluate each model on this difficulty level
        for model_name, model in models.items():
            metrics = evaluate_model(model, data, labels, model_name, f"Synthetic {difficulty}")
            difficulty_results[model_name][difficulty] = metrics
            
            # Plot confusion matrix
            plot_confusion_matrix(
                metrics['confusion_matrix'], 
                model_name, 
                f"Synthetic_{difficulty}"
            )
    
    # Plot performance across difficulty levels
    metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score']
    
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        for model_name in models.keys():
            # Get metric values for each difficulty level
            values = [difficulty_results[model_name][diff][metric] 
                     for diff in difficulties if diff in difficulty_results[model_name]]
            
            # Plot line for this model
            plt.plot(difficulties, values, marker='o', label=model_name)
        
        plt.xlabel('Difficulty Level')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} Across Difficulty Levels')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the figure
        os.makedirs('difficulty_analysis', exist_ok=True)
        plt.savefig(f'difficulty_analysis/{metric}_by_difficulty.png')
        plt.close()
    
    return difficulty_results

def main():
    # Set paths
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'MajorProject')
    original_path = base_path
    synthetic_path = os.path.join(base_path, 'Synthetic')
    
    # Check if paths exist
    if not os.path.exists(original_path):
        print(f"Error: Original data path '{original_path}' not found!")
        return
    
    if not os.path.exists(synthetic_path):
        print(f"Error: Synthetic data path '{synthetic_path}' not found!")
        return
    
    # Load datasets
    print("Loading original dataset...")
    original_data, original_labels = load_data_from_directory(original_path)
    
    print("\nLoading synthetic dataset...")
    synthetic_data, synthetic_labels = load_data_from_directory(synthetic_path)
    
    # Evaluate models on original data
    print("\nEvaluating models on original data...")
    original_results = {}
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        metrics = evaluate_model(model, original_data, original_labels, model_name, "Original")
        original_results[model_name] = metrics
        
        # Plot confusion matrix
        plot_confusion_matrix(metrics['confusion_matrix'], model_name, "Original")
    
    # Evaluate models on synthetic data
    print("\nEvaluating models on synthetic data...")
    synthetic_results = {}
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        metrics = evaluate_model(model, synthetic_data, synthetic_labels, model_name, "Synthetic")
        synthetic_results[model_name] = metrics
        
        # Plot confusion matrix
        plot_confusion_matrix(metrics['confusion_matrix'], model_name, "Synthetic")
    
    # Compare performance
    print("\nComparing performance between original and synthetic data...")
    summary = compare_performance(original_results, synthetic_results)
    print(summary)
    
    # Analyze performance across difficulty levels
    print("\nAnalyzing performance across difficulty levels...")
    difficulty_results = analyze_difficulty_levels(synthetic_path, models)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()