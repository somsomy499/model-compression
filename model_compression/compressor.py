"""Unified compression interface."""
def compress(model, method="quantize", bits=8, **kwargs):
    if method == "quantize":
        from .quantize import Quantizer
        return Quantizer(bits=bits).quantize(model, **kwargs)
    elif method == "prune":
        from .pruner import Pruner
        return Pruner().prune(model, **kwargs)
    elif method == "distill":
        from .distiller import Distiller
        return Distiller().distill(model, **kwargs)
    raise ValueError(f"Unknown method: {method}")
