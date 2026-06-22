# Model Compression Toolkit 🗜️

Comprehensive model compression: pruning, quantization, knowledge distillation, and neural architecture search.

## Methods

| Method | Size Reduction | Speedup | Quality Loss |
|--------|---------------|---------|-------------|
| INT8 Quantization | 4× | 2-3× | <1% |
| Structured Pruning | 2× | 1.8× | <2% |
| Knowledge Distillation | Flexible | 3-5× | 2-5% |
| NAS (efficient arch) | Custom | 5-10× | 1-3% |

## Quick Start

```python
from model_compression import compress

compressed = compress(
    model,
    method="quantize",
    bits=8,
    calibration_data=val_loader,
)
# 4× smaller, 2× faster
```

## License

Apache 2.0