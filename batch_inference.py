# batch_inference.py
import torch
from torchvision import models
import time
from PIL import Image
from torchvision import transforms
import os
import urllib.request


def batch_inference_test():
    print("Testing batch inference performance...")

    # Load pre-trained ResNet model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Create image transform
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load a sample image
    if not os.path.exists("dog.jpg"):
        print("Downloading a sample image...")
        urllib.request.urlretrieve(
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
            "dog.jpg"
        )

    img = Image.open("dog.jpg")
    input_tensor = preprocess(img)

    # Test different batch sizes
    batch_sizes = [1, 8, 16, 32, 64, 128]

    for batch_size in batch_sizes:
        # Create a batch by repeating the same image
        batch = input_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
        
        # Warm-up run
        with torch.no_grad():
            _ = model(batch)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Timed run
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(batch)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        images_per_second = batch_size / total_time
        
        print(f"Batch size: {batch_size}")
        print(f"  Total time: {total_time:.4f} seconds")
        print(f"  Images per second: {images_per_second:.2f}")
        print(f"  Time per image: {(total_time / batch_size) * 1000:.2f} ms")
        
        # GPU memory info if available
        if device.type == "cuda":
            memory_allocated_gb = torch.cuda.memory_allocated() / 1e9
            print(f"  GPU memory used: {memory_allocated_gb:.2f} GB")
        
        print()

if __name__ == "__main__":
    batch_inference_test()
