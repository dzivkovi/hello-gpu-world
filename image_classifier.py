# image_classifier.py
import torch
from torchvision import models
import time
from PIL import Image
from torchvision import transforms
import os
import urllib.request


def classify_image(image_path):
    # Load pre-trained ResNet model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load and preprocess the image
    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0).to(device)

    # Time the inference
    start_time = time.time()

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()

    # Get class prediction
    _, predicted_idx = torch.max(output, 1)

    # Load class labels
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    return {
        "prediction": categories[predicted_idx.item()],
        "time": end_time - start_time,
        "device": device.type
    }

if __name__ == "__main__":
    # Download class names if not present
    if not os.path.exists("imagenet_classes.txt"):
        print("Downloading ImageNet class names...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
            "imagenet_classes.txt"
        )

    # Replace with the path to your image
    image_path = input("Enter the path to an image file (or press Enter for a sample image): ")

    if not image_path or not os.path.exists(image_path):
        if image_path:
            print(f"Image not found: {image_path}")
        # Use a sample image if user's image doesn't exist
        print("Downloading a sample image...")
        urllib.request.urlretrieve(
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
            "dog.jpg"
        )
        image_path = "dog.jpg"
        print(f"Using sample image: {image_path}")

    print(f"\nClassifying image: {image_path}")
    print("Using GPU for inference...")

    result = classify_image(image_path)

    print("\nImage Classification Result:")
    print(f"Prediction: {result['prediction']}")
    print(f"Inference time: {result['time']:.4f} seconds")
    print(f"Using: {result['device'].upper()}")
