"""Post-training quantization calibrator."""
import numpy as np
from typing import Callable, Optional

class Calibrator:
    def __init__(self, num_bins: int = 200, method: str = "minmax"):
        self.num_bins = num_bins
        self.method = method
        self.histograms = {}
        
    def collect(self, name: str, tensor: np.ndarray):
        if name not in self.histograms:
            self.histograms[name] = {"min": float("inf"), "max": float("-inf"), "values": []}
        h = self.histograms[name]
        h["min"] = min(h["min"], float(tensor.min()))
        h["max"] = max(h["max"], float(tensor.max()))
        h["values"].extend(tensor.flatten().tolist())
        
    def compute_ranges(self):
        ranges = {}
        for name, h in self.histograms.items():
            if self.method == "minmax":
                ranges[name] = (h["min"], h["max"])
            elif self.method == "percentile":
                values = np.array(h["values"])
                ranges[name] = (float(np.percentile(values, 1)), float(np.percentile(values, 99)))
        return ranges
