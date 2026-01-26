# CPU Threading Fix for MDX23C Stem Separation

## Problem

When using the MDX23C model for stem separation with CPU inference on macOS, only ~55% CPU utilization was observed, with work being distributed to efficiency cores instead of performance cores.

### Root Cause

PyTorch's default threading configuration was limited to 4 threads, despite the system having 8 cores (4 performance + 4 efficiency cores). This caused:
- Underutilization of available CPU resources
- Work distributed to efficiency cores instead of performance cores
- Slower processing than necessary

### Verification

Before the fix:
```bash
$ conda run -n drumtomidi python -c "import torch; print(torch.get_num_threads())"
4
```

## Solution

Added automatic CPU threading configuration in `OptimizedMDX23CProcessor.__init__()` when `device='cpu'` is used.

### Implementation Details

The `_configure_cpu_threading()` method:

1. **Detects total CPU cores** using `multiprocessing.cpu_count()`
2. **On macOS**, detects performance vs efficiency core split using:
   ```bash
   sysctl -n hw.perflevel0.physicalcpu  # Performance cores
   sysctl -n hw.perflevel1.physicalcpu  # Efficiency cores
   ```
3. **Configures PyTorch threading**:
   - `torch.set_num_threads(cpu_count)` - Uses all available cores
   - `torch.set_num_interop_threads(cpu_count // 2)` - Parallelizes independent operations
4. **Sets environment variables** for underlying libraries:
   - `OMP_NUM_THREADS` - OpenMP threading
   - `MKL_NUM_THREADS` - Intel MKL library threading

### Results

After the fix:
```bash
$ conda run -n drumtomidi python -c "import torch; print('Before:', torch.get_num_threads()); from mdx23c_optimized import OptimizedMDX23CProcessor; p = OptimizedMDX23CProcessor(device='cpu'); print('After:', torch.get_num_threads())"
Before: 4
After: 8
```

**Expected Improvements:**
- CPU utilization should increase from ~55% to close to 100%
- Performance cores should be engaged instead of efficiency cores
- Stem separation processing speed should improve significantly (potentially 2x faster)

## Usage

The fix is automatic - no code changes needed for existing usage:

```python
from mdx23c_optimized import OptimizedMDX23CProcessor

# Threading is automatically configured when device='cpu'
processor = OptimizedMDX23CProcessor(device='cpu')
stems = processor.process_audio('input.wav')
```

## Monitoring CPU Usage

To verify the fix is working, monitor CPU usage during stem separation:

**On macOS:**
```bash
# Open Activity Monitor
# View > CPU Usage > Window CPU Usage
# Look for python process using ~100% CPU
```

**Command line:**
```bash
# While stem separation is running:
top -pid $(pgrep -n python) -stats cpu
```

## Testing

A test script is available to verify threading configuration:

```bash
conda run -n drumtomidi python test_cpu_threading.py
```

This will show:
- Total cores, performance cores, efficiency cores
- PyTorch threading configuration before and after initialization
- Expected CPU utilization percentage

## Files Modified

- `mdx23c_optimized.py` - Added `_configure_cpu_threading()` method
- `agent-plans/bug-tracking.md` - Documented bug and fix
- `docs/CPU_THREADING_FIX.md` - This document

## Related Documentation

- [MDX23C Performance Guide](MDX23C_PERFORMANCE.md)
- [MPS Performance Improvements](MPS_PERFORMANCE_IMPROVEMENTS.md)
- [Mac Native Setup](SETUP_MAC_NATIVE.md)
