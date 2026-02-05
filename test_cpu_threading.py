#!/usr/bin/env python
"""
Test script to verify CPU threading configuration.

Usage:
    conda run -n drumtomidi python test_cpu_threading.py
"""
import torch
import multiprocessing
import os
import subprocess


def get_core_info():
    """Get CPU core information on macOS."""
    info = {
        'total_cores': multiprocessing.cpu_count(),
        'performance_cores': None,
        'efficiency_cores': None
    }
    
    try:
        if os.uname().sysname == 'Darwin':
            # Get performance core count
            result = subprocess.run(
                ['sysctl', '-n', 'hw.perflevel0.physicalcpu'],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0:
                info['performance_cores'] = int(result.stdout.strip())
            
            # Get efficiency core count
            result = subprocess.run(
                ['sysctl', '-n', 'hw.perflevel1.physicalcpu'],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0:
                info['efficiency_cores'] = int(result.stdout.strip())
    except Exception as e:
        print(f"Could not detect core details: {e}")
    
    return info


def main():
    print("=" * 60)
    print("CPU Threading Configuration Test")
    print("=" * 60)
    
    # Get core info
    core_info = get_core_info()
    print(f"\nSystem CPU Information:")
    print(f"  Total cores: {core_info['total_cores']}")
    if core_info['performance_cores']:
        print(f"  Performance cores: {core_info['performance_cores']}")
    if core_info['efficiency_cores']:
        print(f"  Efficiency cores: {core_info['efficiency_cores']}")
    
    # Check current PyTorch settings
    print(f"\nCurrent PyTorch Threading (before OptimizedMDX23CProcessor):")
    print(f"  Intra-op threads: {torch.get_num_threads()}")
    print(f"  Inter-op threads: {torch.get_num_interop_threads()}")
    
    print(f"\nEnvironment Variables:")
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    print(f"  MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'not set')}")
    
    # Initialize processor (which should configure threading)
    print(f"\nInitializing OptimizedMDX23CProcessor with device='cpu'...")
    from mdx23c_optimized import OptimizedMDX23CProcessor
    
    processor = OptimizedMDX23CProcessor(device='cpu')
    
    # Check PyTorch settings after initialization
    print(f"\nPyTorch Threading (after OptimizedMDX23CProcessor):")
    print(f"  Intra-op threads: {torch.get_num_threads()}")
    print(f"  Inter-op threads: {torch.get_num_interop_threads()}")
    
    print(f"\nEnvironment Variables (after):")
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    print(f"  MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'not set')}")
    
    # Calculate expected utilization
    threads = torch.get_num_threads()
    expected_utilization = (threads / core_info['total_cores']) * 100
    
    print(f"\n" + "=" * 60)
    print(f"Expected CPU Utilization: ~{expected_utilization:.0f}%")
    print(f"(with {threads} threads on {core_info['total_cores']} cores)")
    print("=" * 60)
    
    print(f"\nRecommendation:")
    if threads == core_info['total_cores']:
        print("  ✅ Threading is properly configured for full CPU utilization")
    elif threads < core_info['total_cores']:
        print(f"  ⚠️  Underutilizing CPU - using {threads}/{core_info['total_cores']} cores")
    else:
        print(f"  ℹ️  Using {threads} threads on {core_info['total_cores']} cores")


if __name__ == '__main__':
    main()
