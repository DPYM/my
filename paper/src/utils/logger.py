"""
Experiment logging utilities for DualGuard.

Provides:
  - JSON-based experiment result logging
  - Optional TensorBoard integration
  - Consistent experiment naming and directory management
"""

import os
import json
import time
from datetime import datetime


class ExperimentLogger:
    def __init__(self, log_dir="./results", exp_name=None, use_tensorboard=False):
        self.log_dir = log_dir
        self.exp_name = exp_name or datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(log_dir, self.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        self.config = {}
        self.metrics = {}
        self.start_time = time.time()

        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=self.exp_dir)
            except ImportError:
                print("TensorBoard not available. pip install tensorboard")

    def log_config(self, **kwargs):
        self.config.update(kwargs)

    def log_metric(self, key, value, step=None):
        self.metrics[key] = value
        if self.tb_writer is not None and step is not None:
            self.tb_writer.add_scalar(key, value, step)

    def log_metrics(self, metrics_dict, step=None):
        for k, v in metrics_dict.items():
            self.log_metric(k, v, step)

    def save(self):
        elapsed = time.time() - self.start_time
        result = {
            "experiment": self.exp_name,
            "config": self.config,
            "metrics": self.metrics,
            "elapsed_seconds": elapsed,
        }
        path = os.path.join(self.exp_dir, "result.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=float)
        return path

    def close(self):
        path = self.save()
        if self.tb_writer is not None:
            self.tb_writer.close()
        print(f"Results saved to {path}")
