# Using Your NVIDIA GPU for Deep Learning

This guide helps new GPU owners validate their setup and see their GPU in action with PyTorch. You'll run simple benchmarks and a practical image classification example to confirm your GPU is working correctly.

## Requirements

### Hardware

- NVIDIA GPU (tested with RTX 4060)
- 8+ GB RAM recommended

### Software

- Windows 10/11 or Linux
- Python 3.8+ (tested with Python 3.11)
- NVIDIA drivers installed
- CUDA Toolkit (optional for development, but recommended)

## Setup Instructions

### 1. Install NVIDIA Driver and CUDA

First, ensure you have the latest NVIDIA drivers installed. You can check your current installation using:

```bash
nvidia-smi
```

Sample output:

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 572.83                 Driver Version: 572.83         CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 ...  WDDM  |   00000000:01:00.0  On |                  N/A |
| N/A   57C    P8              2W /  132W |     436MiB /   8188MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

For development purposes (compiling CUDA code), install the CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads

After installation, verify CUDA compiler is available:

```bash
nvcc --version
```

Sample output:

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_Pacific_Standard_Time_2023
Cuda compilation tools, release 12.1, V12.1.66
Build cuda_12.1.r12.1/compiler.32415258_0
```

### 2. Create a Python Virtual Environment

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows bash
# OR
.venv\Scripts\activate  # Windows PowerShell
# OR
source .venv/bin/activate  # Linux/Mac
```

### 3. Install PyTorch with CUDA Support

Make sure to install the PyTorch version that matches your CUDA version. For CUDA 12.1:

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
```

For CUDA 12.1+:

```bash
pip install torch torchvision torchaudio
```

### 4. Install Additional Dependencies

```bash
pip install Pillow matplotlib
```

Or create a `requirements.txt` file with:
```
torch==2.4.0+cu121
torchvision>=0.17.0
torchaudio>=2.4.0
pillow>=10.0.0
matplotlib>=3.7.0
```

Then install with:

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121
```

## Validation Scripts

### 1. GPU vs CPU Performance Test

This script compares matrix multiplication performance on GPU vs CPU:

```python
# gpu_vs_cpu.py
import torch
import time
import os

def run_matrix_multiplication(device_name):
    device = torch.device(device_name)
    print(f"\nRunning on {device_name.upper()}...")

    # Create a large tensor
    size = 10000
    print(f"Creating {size}x{size} matrices...")

    # Record start time
    start_time = time.time()

    # Create matrices on the specified device
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Matrix multiplication
    print("Performing matrix multiplication...")
    c = torch.matmul(a, b)

    # Ensure computation is complete
    if device_name == "cuda":
        torch.cuda.synchronize()

    # Record end time
    end_time = time.time()

    elapsed = end_time - start_time
    print(f"Operation completed in {elapsed:.2f} seconds")

    if device_name == "cuda":
        print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return elapsed

def main():
    # Clear the console (works on Windows)
    os.system('cls' if os.name == 'nt' else 'clear')

    print("=" * 60)
    print("GPU vs CPU PERFORMANCE COMPARISON")
    print("=" * 60)

    print("PyTorch version:", torch.__version__)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_time = run_matrix_multiplication("cuda")
    else:
        print("CUDA is not available. Cannot test GPU performance.")
        gpu_time = float('inf')

    # Run the same operation on CPU
    cpu_time = run_matrix_multiplication("cpu")

    # Compare results
    if gpu_time != float('inf'):
        speedup = cpu_time / gpu_time
        print("\n" + "=" * 60)
        print(f"RESULTS: GPU was {speedup:.1f}x faster than CPU!")
        print(f"GPU time: {gpu_time:.2f} seconds")
        print(f"CPU time: {cpu_time:.2f} seconds")
        print("=" * 60)

if __name__ == "__main__":
    main()
```

### 2. Image Classification Demo

This script demonstrates practical GPU usage with a pre-trained ResNet model:

```python
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
```

### 3. Batch Processing Test

This script shows how GPUs excel at processing multiple inputs simultaneously:

```python
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
```

## Running the Tests

After installing the required packages, run each test:

```bash
python gpu_vs_cpu.py
python image_classifier.py
python batch_inference.py
```

## Sample Results

### GPU vs CPU Performance Test

```bash
============================================================
GPU vs CPU PERFORMANCE COMPARISON
============================================================
PyTorch version: 2.4.0+cu121
GPU: NVIDIA GeForce RTX 4060 Laptop GPU

Running on CUDA...
Creating 10000x10000 matrices...
Performing matrix multiplication...
Operation completed in 0.31 seconds
GPU memory used: 1.21 GB

Running on CPU...
Creating 10000x10000 matrices...
Performing matrix multiplication...
Operation completed in 16.82 seconds

============================================================
RESULTS: GPU was 55.0x faster than CPU!
GPU time: 0.31 seconds
CPU time: 16.82 seconds
============================================================
```

### Image Classification Demo

```bash
Enter the path to an image file (or press Enter for a sample image): 
Downloading a sample image...
Using sample image: dog.jpg

Classifying image: dog.jpg
Using GPU for inference...

Image Classification Result:
Prediction: Samoyed
Inference time: 0.1484 seconds
Using: CUDA
```

### Batch Processing Test

```bash
Testing batch inference performance...
Batch size: 1
  Total time: 0.0040 seconds
  Images per second: 250.06
  Time per image: 4.00 ms
  GPU memory used: 0.11 GB

Batch size: 8
  Total time: 0.0122 seconds
  Images per second: 657.35
  Time per image: 1.52 ms
  GPU memory used: 0.12 GB

Batch size: 16
  Total time: 0.0281 seconds
  Images per second: 570.30
  Time per image: 1.75 ms
  GPU memory used: 0.12 GB

Batch size: 32
  Total time: 0.0618 seconds
  Images per second: 518.09
  Time per image: 1.93 ms
  GPU memory used: 0.13 GB

Batch size: 64
  Total time: 0.1185 seconds
  Images per second: 540.31
  Time per image: 1.85 ms
  GPU memory used: 0.15 GB

Batch size: 128
  Total time: 0.2402 seconds
  Images per second: 532.91
  Time per image: 1.88 ms
  GPU memory used: 0.19 GB
```

## Understanding the Results

1. **GPU vs CPU Test**: The matrix multiplication test shows a dramatic 55x speedup on GPU vs CPU for large mathematical operations, which are common in deep learning.

2. **Image Classification**: The ResNet50 model classifies images in ~150ms on GPU, demonstrating practical AI application performance.

3. **Batch Processing**: When processing multiple images at once (batching), the GPU achieves 500+ images per second with minimal memory usage, showing how GPUs excel at parallel processing.

## Troubleshooting

If you encounter issues:

1. **CUDA not available**:
   - Ensure you have the latest NVIDIA drivers installed
   - Verify PyTorch was installed with CUDA support (check `torch.cuda.is_available()`)
   - Try reinstalling PyTorch with the correct CUDA version for your system

2. **NVCC not found**:
   - The CUDA compiler is part of the CUDA Toolkit, not the driver
   - Download and install the CUDA Toolkit from NVIDIA's website
   - Add the CUDA bin directory to your PATH environment variable
   - Note: This is only needed for developing custom CUDA code, not for using PyTorch

3. **Memory errors**:
   - Reduce batch sizes or matrix dimensions
   - Close other GPU-intensive applications
   - For dedicated deep learning, consider a GPU with more VRAM

4. **Import errors**:
   - Ensure all dependencies are installed with `pip install -r requirements.txt`
   - Create a fresh virtual environment if issues persist

## Difference Between NVIDIA Drivers and CUDA Toolkit

- **NVIDIA Drivers**: Required for basic GPU operation; allows PyTorch to use the GPU
- **CUDA Toolkit**: Optional development tools for writing custom CUDA code (includes nvcc compiler)

Most users only need the drivers for running pre-built models in PyTorch. The CUDA Toolkit is necessary only if you're developing custom CUDA extensions.

## Next Steps

Now that you've confirmed your GPU is working properly, you can:

1. Try more advanced deep learning models like YOLO object detection or Stable Diffusion
2. Train your own neural networks
3. Explore GPU-accelerated data processing with libraries like RAPIDS

Congratulations on successfully setting up and testing your GPU for deep learning!
