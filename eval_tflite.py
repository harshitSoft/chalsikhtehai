import cv2
import numpy as np
import string
import tensorflow as tf
import argparse
import time
from tqdm import tqdm
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description="Evaluate TFLite OCR model or run inference on a single image.")
parser.add_argument('--eval_dir', type=str, default='data/ocr_training_data', help='Directory with evaluation images and gt.txt')
parser.add_argument('--image', type=str, help='Path to a single image to run inference')
parser.add_argument('--quantization', type=str, default='float16', choices=['float16'], help='Quantization type of TFLite model (default: float16)')

# Define character set (digits, lowercase letters, and decimal point)
alphabet = string.digits + string.ascii_lowercase + '.'
blank_index = len(alphabet)

def run_tflite_model(image_path, interpreter):
    """Load image, preprocess, run through TFLite model, and return raw output."""
    input_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if input_data is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    input_data = cv2.resize(input_data, (200, 31))
    input_data = input_data[np.newaxis, :, :, np.newaxis].astype('float32') / 255.0

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    return output

def decode_output(tflite_output):
    """Convert raw output into readable string using greedy decoding."""
    output_sequence = tflite_output[0]
    prediction = []
    previous = -1

    for index in output_sequence:
        if index == previous:
            continue
        if index == blank_index or index == -1:
            previous = index
            continue
        prediction.append(alphabet[index])
        previous = index

    decoded_text = ''.join(prediction)

    # Convert to float if applicable
    try:
        if '.' in decoded_text:
            decoded_text = str(float(decoded_text))
    except ValueError:
        pass

    return decoded_text

def evaluate_dataset(model_path, data_dir):
    """Evaluate model on full dataset in directory."""
    gt_file = os.path.join(data_dir, 'gt.txt')
    with open(gt_file) as file:
        lines = file.read().splitlines()
        dataset = [(os.path.join(data_dir, line.split('\t')[0]), line.split('\t')[1]) for line in lines]

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    predictions = []
    with tqdm(total=len(dataset), desc="Evaluating") as pbar:
        for image_path, _ in dataset:
            output = run_tflite_model(image_path, interpreter)
            prediction = decode_output(output)
            predictions.append(prediction)
            pbar.update()

    return dataset, predictions

def evaluate_accuracy(dataset, predictions):
    """Compare predictions to ground truth and calculate accuracy."""
    results_boolean = []

    for (_, actual), predicted in zip(dataset, predictions):
        try:
            if '.' in predicted:
                actual_float = float(actual)
                predicted_float = float(predicted)
                results_boolean.append(abs(actual_float - predicted_float) < 0.01)
            else:
                results_boolean.append(actual == predicted)
        except ValueError:
            results_boolean.append(actual == predicted)

    accuracy = round(100 * sum(results_boolean) / len(results_boolean), 2)
    return accuracy

def main():
    args = parser.parse_args()
    model_path = f"meeter_rec_{args.quantization}.tflite"

    if args.image:
        print(f"\nRunning inference on single image: {args.image}")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        output = run_tflite_model(args.image, interpreter)
        prediction = decode_output(output)
        print(f"Predicted text: {prediction}")

    else:
        print(f"\nEvaluating full dataset in directory: {args.eval_dir}")
        start_time = time.time()

        dataset, predictions = evaluate_dataset(model_path, args.eval_dir)
        accuracy = evaluate_accuracy(dataset, predictions)

        avg_inference_time = round((time.time() - start_time) / len(dataset), 2)

        print(f"\nEvaluation Results:")
        print(f"    Accuracy: {accuracy}%")
        print(f"    Average inference time: {avg_inference_time} sec/image")

if __name__ == "__main__":
    main()
