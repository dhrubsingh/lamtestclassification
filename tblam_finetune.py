import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

# Set fixed random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Data loading settings
WIDTH = 224
HEIGHT = 224
CHANNEL = 3

def load_data_from_directory(data_path):
    """Load and prepare dataset from a directory with three classes"""
    # Get file lists for each class
    neg_images = os.listdir(os.path.join(data_path, 'Negative'))
    pos_images = os.listdir(os.path.join(data_path, 'Positive'))
    ind_images = os.listdir(os.path.join(data_path, 'Indeterminant'))
    
    # Create dataframes for each class
    neg_df = pd.DataFrame({'id': neg_images, 'label': 'Negative'})
    pos_df = pd.DataFrame({'id': pos_images, 'label': 'Positive'})
    ind_df = pd.DataFrame({'id': ind_images, 'label': 'Indeterminant'})
    
    data = []
    labels = []
    filenames = []
    
    # Load negative images
    for img_id in neg_df['id']:
        img_path = os.path.join(data_path, 'Negative', img_id)
        try:
            img = Image.open(img_path)
            img = img.convert('RGB')  # Ensure RGB format
            img = np.array(img.resize((WIDTH, HEIGHT))) / 255.0
            data.append(img)
            labels.append('Negative')
            filenames.append(img_id)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    # Load positive images
    for img_id in pos_df['id']:
        img_path = os.path.join(data_path, 'Positive', img_id)
        try:
            img = Image.open(img_path)
            img = img.convert('RGB')  # Ensure RGB format
            img = np.array(img.resize((WIDTH, HEIGHT))) / 255.0
            data.append(img)
            labels.append('Positive')
            filenames.append(img_id)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    # Load indeterminant images
    for img_id in ind_df['id']:
        img_path = os.path.join(data_path, 'Indeterminant', img_id)
        try:
            img = Image.open(img_path)
            img = img.convert('RGB')  # Ensure RGB format
            img = np.array(img.resize((WIDTH, HEIGHT))) / 255.0
            data.append(img)
            labels.append('Indeterminant')
            filenames.append(img_id)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    data = np.array(data)
    labels = np.array(labels)
    
    # Convert string labels to numeric using LabelEncoder
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)
    
    print(f"Loaded {len(data)} images from {data_path}")
    print(f"Class distribution:")
    for i, label in enumerate(label_encoder.classes_):
        count = np.sum(numeric_labels == i)
        print(f"  {label}: {count}")
    
    return data, numeric_labels, filenames, label_encoder

def evaluate_model(model, X, y, label_encoder, model_name, dataset_name):
    """Evaluate a model and return metrics"""
    # Get predictions
    pred = model.predict(X, verbose=0)
    y_pred = np.argmax(pred, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    
    # Calculate per-class metrics
    class_names = label_encoder.classes_
    metrics_dict = {'accuracy': accuracy}
    
    for i, class_name in enumerate(class_names):
        y_true_binary = (y == i)
        y_pred_binary = (y_pred == i)
        
        # Use zero_division=0 to handle cases where a class has no predicted samples
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        print(f"\n{model_name} on {dataset_name} - Class {class_name}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        metrics_dict[f'{class_name}_precision'] = precision
        metrics_dict[f'{class_name}_recall'] = recall
        metrics_dict[f'{class_name}_f1'] = f1
    
    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred)
    print(f"\nConfusion Matrix:")
    print("Predicted →")
    print("             Neg   Pos   Ind")
    print(f"Actual Neg: {cm[0]}")
    print(f"      Pos: {cm[1]}")
    print(f"      Ind: {cm[2]}")
    
    metrics_dict['confusion_matrix'] = cm
    return metrics_dict

def create_model_a():
    """Create Model A - DenseNet121 based"""
    base_model = tf.keras.applications.DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(HEIGHT, WIDTH, CHANNEL)
    )
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model

def create_model_b():
    """Create Model B - ResNet50V2 based"""
    base_model = tf.keras.applications.ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(HEIGHT, WIDTH, CHANNEL)
    )
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model

def create_model_c():
    """Create Model C - NASNetMobile based"""
    base_model = tf.keras.applications.NASNetMobile(
        weights='imagenet',
        include_top=False,
        input_shape=(HEIGHT, WIDTH, CHANNEL)
    )
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model

def create_model_d():
    """Create Model D - EfficientNetB0 based"""
    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(HEIGHT, WIDTH, CHANNEL)
    )
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model

def fine_tune_model(model, model_name, X_train, y_train, X_val, y_val, epochs=20, batch_size=32, class_weight_param=None):
    """Fine-tune a model on new data"""
    print(f"\nFine-tuning {model_name}...")
    
    # Create a new model with the same architecture
    fine_tuned_model = tf.keras.models.clone_model(model)
    fine_tuned_model.set_weights(model.get_weights())
    
    # Unfreeze some layers for fine-tuning
    base_model = fine_tuned_model.layers[0]
    base_model.trainable = True
    
    # For Model D, unfreeze more layers for better adaptation
    num_layers_to_freeze = 30
    if model_name == "Model_D":
        num_layers_to_freeze = 50
    for layer in base_model.layers[:-num_layers_to_freeze]:
        layer.trainable = False
    
    # Compile the model
    fine_tuned_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001
    )
    
    # Create directory for model checkpoints
    os.makedirs('fine_tuned_models', exist_ok=True)
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f'fine_tuned_models/{model_name}_fine_tuned.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train the model with optional class weights
    history = fine_tuned_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        class_weight=class_weight_param,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    
    # Create directory for training plots
    os.makedirs('training_plots', exist_ok=True)
    plt.savefig(f'training_plots/{model_name}_fine_tuning_history.png')
    plt.close()
    
    return fine_tuned_model

def main():
    # Set paths
    # original dataset
    #data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dataset')
    
    # preprocessed dataset
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Preprocessed')    

    # Check if path exists
    if not os.path.exists(data_path):
        print(f"Error: Data path '{data_path}' not found!")
        return
    
    # Load dataset
    print("Loading TB LAM dataset...")
    data, labels, filenames, label_encoder = load_data_from_directory(data_path)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Further split training data to create a validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    print("\nDataset splits:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Compute class weights for Model D
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print("\nClass weights for Model D:", class_weight_dict)
    
    # Create and train all models
    models = {
        'Model_A': create_model_a(),
        'Model_B': create_model_b(),
        'Model_C': create_model_c(),
        'Model_D': create_model_d()
    }
    
    # Results dictionary to store metrics
    results = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training and evaluating {model_name}")
        print(f"{'='*50}")
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # If the model is Model_D, use class weights during training
        fit_class_weight = class_weight_dict if model_name == "Model_D" else None
        
        # Train base model
        print(f"\nTraining base {model_name}...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=32,
            class_weight=fit_class_weight,
            verbose=1
        )
        
        # Evaluate base model
        print(f"\nEvaluating base {model_name}...")
        base_results = evaluate_model(model, X_test, y_test, label_encoder, f"{model_name}", "Test Set")
        results[f"{model_name}_base"] = base_results
        
        # Fine-tune the model
        fine_tuned_model = fine_tune_model(
            model, model_name,
            X_train, y_train,
            X_val, y_val,
            epochs=20,
            batch_size=32,
            class_weight_param=fit_class_weight
        )
        
        # Evaluate fine-tuned model
        print(f"\nEvaluating fine-tuned {model_name}...")
        fine_tuned_results = evaluate_model(fine_tuned_model, X_test, y_test, label_encoder, f"{model_name} (Fine-tuned)", "Test Set")
        results[f"{model_name}_fine_tuned"] = fine_tuned_results
        
        # Save the final model (using .keras extension)
        os.makedirs('fine_tuned_models', exist_ok=True)
        fine_tuned_model.save(f'fine_tuned_models/{model_name}_final.keras')
        print(f"\nFinal {model_name} saved as 'fine_tuned_models/{model_name}_final.keras'")
    
    # Save results to CSV
    results_list = []
    for model_version, metrics in results.items():
        model_metrics = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
        results_list.append({
            'Model': model_version,
            **model_metrics
        })
    
    # Create DataFrame from the list of dictionaries
    results_df = pd.DataFrame(results_list)
    
    # Save to CSV
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("\nResults saved to 'model_comparison_results.csv'")

if __name__ == "__main__":
    main()
