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
    """
    Load and prepare dataset from a directory structured with three subdirectories:
    'Negative', 'Positive', and 'Indeterminant'.
    """
    # List image files in each subdirectory
    neg_images = os.listdir(os.path.join(data_path, 'Negative'))
    pos_images = os.listdir(os.path.join(data_path, 'Positive'))
    ind_images = os.listdir(os.path.join(data_path, 'Indeterminant'))
    
    data = []
    labels = []
    filenames = []
    
    # Helper function to load images from a folder and append data/labels
    def load_images(folder_name, label_str, image_list):
        for img_id in image_list:
            img_path = os.path.join(data_path, folder_name, img_id)
            try:
                img = Image.open(img_path)
                img = img.convert('RGB')
                img = np.array(img.resize((WIDTH, HEIGHT))) / 255.0
                data.append(img)
                labels.append(label_str)
                filenames.append(img_id)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    load_images('Negative', 'Negative', neg_images)
    load_images('Positive', 'Positive', pos_images)
    load_images('Indeterminant', 'Indeterminant', ind_images)
    
    data = np.array(data)
    labels = np.array(labels)
    
    # Convert string labels to numeric
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(labels)
    
    print(f"Loaded {len(data)} images from {data_path}")
    print("Class distribution:")
    for i, label in enumerate(label_encoder.classes_):
        count = np.sum(numeric_labels == i)
        print(f"  {label}: {count}")
    
    return data, numeric_labels, filenames, label_encoder

def evaluate_model(model, X, y, label_encoder, dataset_name):
    """
    Evaluate the model on data X, y and print per-class metrics plus a confusion matrix.
    Returns a dictionary of metrics.
    """
    pred = model.predict(X, verbose=0)
    y_pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y, y_pred)
    class_names = label_encoder.classes_
    metrics_dict = {'accuracy': accuracy}
    
    for i, class_name in enumerate(class_names):
        y_true_binary = (y == i)
        y_pred_binary = (y_pred == i)
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        print(f"\nModel D on {dataset_name} - Class {class_name}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        metrics_dict[f'{class_name}_precision'] = precision
        metrics_dict[f'{class_name}_recall'] = recall
        metrics_dict[f'{class_name}_f1'] = f1
    
    cm = confusion_matrix(y, y_pred)
    print("\nConfusion Matrix:")
    print("Predicted →")
    print("             Neg   Pos   Ind")
    print(f"Actual Neg: {cm[0]}")
    print(f"      Pos: {cm[1]}")
    print(f"      Ind: {cm[2]}")
    metrics_dict['confusion_matrix'] = cm
    return metrics_dict

def create_model_d():
    """
    Create Model D based on EfficientNetB0 with modifications for improved training:
      - Lower dropout rate.
      - Added BatchNormalization.
    """
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
        tf.keras.layers.Dropout(0.2),  # Lower dropout rate
        tf.keras.layers.BatchNormalization(),  # Stabilize training
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model

def fine_tune_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32, class_weight_param=None):
    """
    Fine-tune Model D.
    For Model D, we unfreeze most of the base model (freeze only the first 10 layers)
    and use a higher learning rate.
    """
    print("\nFine-tuning Model D...")
    fine_tuned_model = tf.keras.models.clone_model(model)
    fine_tuned_model.set_weights(model.get_weights())
    
    base_model = fine_tuned_model.layers[0]
    base_model.trainable = True
    # Unfreeze almost all layers; freeze only the first 10 layers.
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    
    # Use a higher learning rate for fine-tuning Model D
    fine_tuned_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-5
    )
    
    os.makedirs('fine_tuned_models', exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'fine_tuned_models/Model_D_fine_tuned.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
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
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title("Model D Fine-Tuning - Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title("Model D Fine-Tuning - Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    os.makedirs('training_plots', exist_ok=True)
    plt.savefig('training_plots/Model_D_fine_tuning_history.png')
    plt.close()
    
    return fine_tuned_model

def main():
    # Set the dataset path (update this path as needed)
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dataset')
    if not os.path.exists(data_path):
        print(f"Error: Data path '{data_path}' not found!")
        return
    
    print("Loading TB LAM dataset...")
    data, labels, filenames, label_encoder = load_data_from_directory(data_path)
    
    # Split data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, stratify=labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    print("\nDataset splits:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Compute class weights (Model D is sensitive to class imbalance)
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print("\nClass weights for Model D:", class_weight_dict)
    
    # Create Model D
    model_d = create_model_d()
    
    # Compile base model
    model_d.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the base version of Model D using class weights
    print("\nTraining base Model D...")
    history = model_d.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    print("\nEvaluating base Model D...")
    base_results = evaluate_model(model_d, X_test, y_test, label_encoder, "Test Set")
    
    # Fine-tune Model D
    fine_tuned_model_d = fine_tune_model(model_d, X_train, y_train, X_val, y_val,
                                           epochs=20, batch_size=32,
                                           class_weight_param=class_weight_dict)
    
    print("\nEvaluating fine-tuned Model D...")
    fine_tuned_results = evaluate_model(fine_tuned_model_d, X_test, y_test, label_encoder, "Test Set")
    
    # Save the final fine-tuned model
    os.makedirs('fine_tuned_models', exist_ok=True)
    fine_tuned_model_d.save('fine_tuned_models/Model_D_final.keras')
    print("\nFinal Model D saved as 'fine_tuned_models/Model_D_final.keras'")
    
    # Save the evaluation results to a CSV file
    results = {
        'Base': base_results,
        'Fine_Tuned': fine_tuned_results
    }
    results_list = []
    for version, metrics in results.items():
        metrics_filtered = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
        results_list.append({
            'Model': f"Model_D_{version}",
            **metrics_filtered
        })
    results_df = pd.DataFrame(results_list)
    results_df.to_csv('Model_D_results.csv', index=False)
    print("\nResults saved to 'Model_D_results.csv'")

if __name__ == "__main__":
    main()
