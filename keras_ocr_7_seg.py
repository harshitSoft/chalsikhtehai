import os
import random
import string
import itertools
import numpy as np
import tensorflow as tf
import sklearn.model_selection
import keras_ocr
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics import confusion_matrix
from matplotlib.pyplot import figure
from imgaug import augmenters as iaa


# Constants and Config
DATA_DIR = './ocr_training_data'
GT_FILE = os.path.join(DATA_DIR, 'gt.txt')
ALPHABET = string.digits + '.'
BATCH_SIZE = 8
EPOCHS = 250
MODEL_FILENAME = 'meeter_rec.h5'
CSV_LOG_FILE = 'meeter_rec.csv'
TFLITE_QUANTIZATION = ['float16']
TFLITE_FILENAME_TEMPLATE = 'meeter_rec_{}.tflite'


# Load training labels
def load_labels(gt_path, base_dir):
    with open(gt_path) as file:
        lines = file.read().splitlines()
    return [(os.path.join(base_dir, line.split('\t')[0]), None, line.split('\t')[1]) for line in lines]


# Resize training images to model input shape
def resize_images(directory, width, height):
    for item in os.listdir(directory):
        if item.endswith('.png'):
            image_path = os.path.join(directory, item)
            with Image.open(image_path) as img:
                resized = img.resize((width, height), Image.BICUBIC)
                resized.save(image_path, 'PNG')


# Image augmentation pipeline
def build_augmenter():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    return iaa.Sequential(
        [
            sometimes(iaa.Affine(
                scale={"x": (0.1, 1.1), "y": (0.1, 1.1)},
                rotate=(-10, 10),
                shear=(-8, 8),
                order=[0, 1],
                cval=(0, 255),
                mode='constant'
            )),
            iaa.SomeOf((0, 3), [
                iaa.OneOf([
                    iaa.GaussianBlur((0, 1.0)),
                    iaa.AverageBlur(k=(1, 3)),
                    iaa.MedianBlur(k=(1, 3)),
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                sometimes(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0, 0.25)),
                    iaa.DirectedEdgeDetect(alpha=(0, 0.5), direction=(0.0, 0.2)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03*255), per_channel=0.5),
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True),
                iaa.Add((-10, 10), per_channel=0.5),
                iaa.Multiply((0.7, 1.2), per_channel=0.5),
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                sometimes(iaa.JpegCompression(compression=(70, 99)))
            ], random_order=True)
        ],
        random_order=True
    )


# Train the recognizer
def train_recognizer():
    labels = load_labels(GT_FILE, DATA_DIR)
    print(f'Total samples: {len(labels)}')

    recognizer = keras_ocr.recognition.Recognizer(alphabet=ALPHABET)
    recognizer.compile()
    h, w = recognizer.model.input_shape[1:3]

    resize_images(DATA_DIR, w, h)
    augmenter = build_augmenter()

    train_labels, val_labels = sklearn.model_selection.train_test_split(labels, test_size=0.2, random_state=42)

    (train_img_gen, train_steps), (val_img_gen, val_steps) = [
        (
            keras_ocr.datasets.get_recognizer_image_generator(
                labels=labs,
                height=h,
                width=w,
                alphabet=ALPHABET,
                augmenter=aug
            ),
            len(labs) // BATCH_SIZE
        )
        for labs, aug in [(train_labels, augmenter), (val_labels, None)]
    ]

    train_gen, val_gen = [
        recognizer.get_batch_generator(img_gen, BATCH_SIZE)
        for img_gen in [train_img_gen, val_img_gen]
    ]

    # Show one image and label
    img, txt = next(train_img_gen)
    print('Sample text:', txt)
    plt.imshow(img)
    plt.show()

    history = recognizer.training_model.fit(
        x=train_gen,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        validation_data=val_gen,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(MODEL_FILENAME, monitor='loss', save_best_only=True),
            tf.keras.callbacks.CSVLogger(CSV_LOG_FILE)
        ],
        epochs=EPOCHS,
        verbose=1
    )

    # Plot training history
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return recognizer, val_labels


# Evaluate the recognizer
def evaluate_model(recognizer, val_labels):
    figure(figsize=(16, 12), dpi=120)
    correct, all_labels, all_preds = [], [], []
    i = 0

    for image_path, _, actual in val_labels:
        predicted = recognizer.recognize(image_path)
        try:
            if '.' in predicted:
                predicted = str(float(predicted))
        except ValueError:
            pass

        correct.append(predicted == actual)

        if len(actual) == len(predicted):
            all_labels.extend(actual)
            all_preds.extend(predicted)

        if predicted != actual and i < 32:
            img = Image.open(image_path)
            plt.subplot(8, 4, i + 1)
            plt.title(f'Pred: {predicted}, Actual: {actual}')
            plt.axis('off')
            plt.imshow(img)
            i += 1

    accuracy = sum(correct) / len(correct)
    print(f'Validation Accuracy: {accuracy:.4f}')
    return all_labels, all_preds


# Plot confusion matrix
def plot_confusion_matrix(cm, labels, normalize=True, title='Confusion Matrix'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    plt.figure(figsize=(12, 8), dpi=100)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    if normalize:
        cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]:.0f}" if normalize else f"{cm[i, j]:,}",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel(f'Predicted label\naccuracy={accuracy:.4f}; misclass={misclass:.4f}')
    plt.ylabel('True label')
    plt.show()


# Convert model to TFLite
def export_tflite_model(model, quant_types):
    for qt in quant_types:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
        ]

        if qt == 'float16':
            converter.target_spec.supported_types = [tf.float16]
        elif qt == 'full_int8':
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        tflite_model = converter.convert()
        tflite_path = TFLITE_FILENAME_TEMPLATE.format(qt)
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f'Saved TFLite model: {tflite_path}')


# Main Execution
if __name__ == '__main__':
    recognizer, val_labels = train_recognizer()
    true_labels, pred_labels = evaluate_model(recognizer, val_labels)
    target_chars = list(string.digits) + ['.']
    cm = confusion_matrix(true_labels, pred_labels, labels=target_chars)
    plot_confusion_matrix(cm, target_chars)
    export_tflite_model(recognizer.prediction_model, TFLITE_QUANTIZATION)
