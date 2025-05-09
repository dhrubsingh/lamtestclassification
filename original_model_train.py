import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten
from tensorflow.keras.applications import NASNetMobile, DenseNet121, ResNet50


"""
Please note that this script has been adapated from the original Jupyter Notebook to create the original models
locally.
"""

# Confiugration
WIDTH = 224
HEIGHT = 224
CHANNEL = 3
train_path = '/root/covid-classification/MajorProject'


def load_data():
    # load positive and negative images
    neg_images_one = os.listdir(train_path + '/Negative')
    neg_df_one = pd.DataFrame({'id': neg_images_one})
    neg_df_one['label'] = 0
    neg_ids_one = neg_df_one['id']
    post_images = os.listdir(train_path + '/Positive')
    post_df = pd.DataFrame({'id': post_images})
    post_df['label'] = 1
    post_ids = post_df['id']
    
    
    # process images
    data_neg_one = []
    for i in range(len(neg_df_one)):
        image = Image.open(train_path + '/Negative/' + neg_ids_one[i])
        image = np.array(image.resize((WIDTH, HEIGHT)))
        data_neg_one.append(image)
    data_neg_one = np.array(data_neg_one)

    post_data = []
    for i in range(len(post_df)):
        image = Image.open(train_path + '/Positive/' + post_ids[i])
        image = np.array(image.resize((WIDTH, HEIGHT)))
        post_data.append(image)
    post_data = np.array(post_data)
    
    # label data
    data = np.concatenate((data_neg_one, post_data))
    frames = [neg_df_one, post_df]
    df = pd.concat(frames)
    labels = df['label']
    
    # preprocess and convert data
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)
    X_train = X_train / 255
    X_test = X_test / 255
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    
    return X_train, X_test, y_train, y_test

# Model A - Default CNN with Adam optimizer
def train_model_a(X_train, X_test, y_train, y_test):
    print("\n=== Training Model A: Default CNN with Adam optimizer ===")
    
    # build and compile model
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(HEIGHT, WIDTH, CHANNEL)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    opt = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    
    # train model
    monitor_val_acc = EarlyStopping(monitor='val_loss', patience=15)
    checkpoint = ModelCheckpoint('my_model.h5', verbose=1, save_best_only=True)
    
    history = model.fit(
        X_train, y_train, 
        batch_size=30, 
        epochs=100, 
        validation_data=(X_test, y_test), 
        callbacks=[checkpoint, monitor_val_acc]
    )
    
    # evaluate and plot model
    evaluation = model.evaluate(X_test, y_test)
    print(f"Model A - Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")
    plot_history(history, "Model A")
    
    gc.collect()
    
    return model

# Model B - Default CNN with SGD optimizer
def train_model_b(X_train, X_test, y_train, y_test):
    print("\n=== Training Model B: Default CNN with SGD optimizer ===")
    
    # build and compile model
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(HEIGHT, WIDTH, CHANNEL)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    
    # Compile model
    opt = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    
    # train model
    monitor_val_acc = EarlyStopping(monitor='val_loss', patience=15)
    checkpoint = ModelCheckpoint('my_model-SGD.h5', verbose=1, save_best_only=True)  
    history = model.fit(
        X_train, y_train, 
        batch_size=50, 
        epochs=100, 
        validation_data=(X_test, y_test), 
        callbacks=[checkpoint, monitor_val_acc]
    )
    
    # evaluate and plot model
    evaluation = model.evaluate(X_test, y_test)
    print(f"Model B - Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")
    plot_history(history, "Model B")

    gc.collect()
    
    return model

# Model C - NASNetMobile
def train_model_c(X_train, X_test, y_train, y_test):
    print("\n=== Training Model C: NASNetMobile ===")
    
    # build model
    model_d = NASNetMobile(weights='imagenet', include_top=False, input_shape=(WIDTH, HEIGHT, CHANNEL))
    x = model_d.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    preds = Dense(2, activation='softmax')(x)
    model = Model(inputs=model_d.input, outputs=preds)
    
    # freeze early layers and compile model
    for layer in model.layers[:-8]:
        layer.trainable = False
    
    for layer in model.layers[-8:]:
        layer.trainable = True
    
    opt = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    
    redlr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, verbose=1, min_lr=1e-2)
    checkpoint = ModelCheckpoint('model-nasnet.h5', verbose=1, save_best_only=True)
    monitor_val_acc = EarlyStopping(monitor='val_loss', patience=15)
    
    # data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        zoom_range=0.2,
        horizontal_flip=True,
        shear_range=0.2,
        vertical_flip=True,
        rotation_range=10,
        fill_mode="nearest"
    )
    datagen.fit(X_train)
    
    # train and evaluate model
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=50),
        epochs=100,
        callbacks=[redlr, checkpoint, monitor_val_acc],
        validation_data=(X_test, y_test)
    )
    
    model = tf.keras.models.load_model('model-nasnet.h5')
    
    plot_history(history, "Model C - NASNetMobile")
    
    gc.collect()
    
    return model

# Model D - DenseNet121
def train_model_d(X_train, X_test, y_train, y_test):
    print("\n=== Training Model D: DenseNet121 ===")
    
    # build model and freeze early layers
    model_d = DenseNet121(weights='imagenet', include_top=False, input_shape=(WIDTH, HEIGHT, CHANNEL))
    x = model_d.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    preds = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=model_d.input, outputs=preds)
    
    for layer in model.layers[:-8]:
        layer.trainable = False
    
    for layer in model.layers[-8:]:
        layer.trainable = True
    
    # complie model and perform data augmentation
    opt = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    redlr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-2)
    checkpoint = ModelCheckpoint('densenet-model.h5', verbose=1, save_best_only=True)
    monitor_val_acc = EarlyStopping(monitor='val_loss', patience=15)
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        zoom_range=0.2,
        horizontal_flip=True,
        shear_range=0.2,
        vertical_flip=True,
        rotation_range=10,
        fill_mode="nearest"
    )
    datagen.fit(X_train)
    
    # train model and plot results
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=50),
        epochs=100,
        callbacks=[redlr, checkpoint, monitor_val_acc],
        validation_data=(X_test, y_test)
    )

    model = tf.keras.models.load_model('densenet-model.h5')
    
    plot_history(history, "Model D - DenseNet121")
    
    gc.collect()
    
    return model

# Model E - ResNet50
def train_model_e(X_train, X_test, y_train, y_test):
    print("\n=== Training Model E: ResNet50 ===")
    
    # build model and freeze early layers
    model_d = ResNet50(weights='imagenet', include_top=False, input_shape=(WIDTH, HEIGHT, CHANNEL))
    x = model_d.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    preds = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=model_d.input, outputs=preds)
    
    for layer in model.layers[:-8]:
        layer.trainable = False
    
    for layer in model.layers[-8:]:
        layer.trainable = True
    
    # compile model and perform data augmentation
    opt = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    
    redlr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-2)
    checkpoint = ModelCheckpoint('resnet50-model.h5', verbose=1, save_best_only=True)
    monitor_val_acc = EarlyStopping(monitor='val_loss', patience=15)
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        zoom_range=0.2,
        horizontal_flip=True,
        shear_range=0.2,
        vertical_flip=True,
        rotation_range=10,
        fill_mode="nearest"
    )
    datagen.fit(X_train)
    
    # train model and plot results
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=50),
        epochs=100,
        callbacks=[redlr, checkpoint, monitor_val_acc],
        validation_data=(X_test, y_test)
    )
    
    plot_history(history, "Model E - ResNet50")
    gc.collect()
    
    return model

# help plot training history
def plot_history(history, model_name):
    # accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    
    # loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Data shapes - X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")
    
    # train all the models
    model_a = train_model_a(X_train, X_test, y_train, y_test)
    model_b = train_model_b(X_train, X_test, y_train, y_test)
    model_c = train_model_c(X_train, X_test, y_train, y_test)
    model_d = train_model_d(X_train, X_test, y_train, y_test)
    model_e = train_model_e(X_train, X_test, y_train, y_test)
    
    print("\nTraining complete. Model files saved:")
    print("- Model A: my_model.h5")
    print("- Model B: my_model-SGD.h5")
    print("- Model C: model-nasnet.h5")
    print("- Model D: densenet-model.h5")
    print("- Model E: resnet50-model.h5")

if __name__ == "__main__":
    main()