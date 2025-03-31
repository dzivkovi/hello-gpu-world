import os
import sys
import warnings
import contextlib

# Check NumPy version before importing torch
try:
    import numpy as np
    numpy_version = np.__version__
    if numpy_version.startswith('2.'):
        print(f"NOTE: Using NumPy {numpy_version}. For cleaner output, consider: pip install numpy==1.26.4")
except ImportError:
    pass

# Create a context manager to capture and filter stderr/stdout
@contextlib.contextmanager
def suppress_output():
    # Save original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Create a null device to redirect output
    null_device = open(os.devnull, 'w')
    
    try:
        # Redirect stdout and stderr to the null device
        sys.stdout = null_device
        sys.stderr = null_device
        
        # Also silence Python warnings
        warnings.filterwarnings("ignore")
        
        yield
    finally:
        # Restore original stdout, stderr and warning settings
        sys.stdout = original_stdout  
        sys.stderr = original_stderr
        warnings.resetwarnings()
        null_device.close()

# Import torch with all output suppressed
print("Loading PyTorch (suppressing compatibility messages)...")
with suppress_output():
    import torch
print("PyTorch loaded successfully!")

import time
import gc

def format_size(size_bytes):
    """Format bytes to a human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0 or unit == 'TB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

def run_benchmark(n=10000):
    """Run a matrix multiplication benchmark comparing GPU vs CPU performance"""
    print("="*60)
    print("GPU vs CPU PERFORMANCE COMPARISON")
    print("="*60)
    
    # Print system information
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your installation.")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    
    # GPU benchmark
    print("Running on CUDA...")
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Creating {n}x{n} matrices...")
    # Measure GPU memory before
    start_mem = torch.cuda.memory_allocated(0)
    
    # Create matrices on GPU
    start_time = time.time()
    a = torch.randn(n, n, device='cuda')
    b = torch.randn(n, n, device='cuda')
    
    print("Performing matrix multiplication...")
    # Perform multiplication and synchronize to ensure timing is accurate
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    gpu_time = time.time() - start_time
    print(f"Operation completed in {gpu_time:.2f} seconds")
    
    # Measure GPU memory after
    end_mem = torch.cuda.memory_allocated(0)
    mem_used = end_mem - start_mem
    print(f"GPU memory used: {format_size(mem_used)}")
    
    # Free GPU memory
    del a, b, c
    torch.cuda.empty_cache()
    gc.collect()
    
    # CPU benchmark
    print("Running on CPU...")
    print(f"Creating {n}x{n} matrices...")
    
    start_time = time.time()
    a = torch.randn(n, n)
    b = torch.randn(n, n)
    
    print("Performing matrix multiplication...")
    c = torch.matmul(a, b)
    
    cpu_time = time.time() - start_time
    print(f"Operation completed in {cpu_time:.2f} seconds")
    
    # Cleanup CPU tensors
    del a, b, c
    gc.collect()
    
    # Show comparison
    print("="*60)
    speedup = cpu_time / gpu_time
    print(f"RESULTS: GPU was {speedup:.1f}x faster than CPU!")
    print(f"GPU time: {gpu_time:.2f} seconds")
    print(f"CPU time: {cpu_time:.2f} seconds")
    print("="*60)

if __name__ == "__main__":
    run_benchmark()
