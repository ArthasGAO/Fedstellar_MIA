import json
import os
from lightning import Callback
import torch

class MyCustomCheckpoint(Callback):
    def __init__(self, save_dir, idx, logger):
        super().__init__()
        self.save_dir = save_dir
        self.idx = idx
        self.logger = logger
        
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        res_dict = {key: value.item() if hasattr(value, 'item') else value for key, value in metrics.items()}
        self.logger.info(f"Client {self.idx} Local Model training result: {json.dumps(res_dict, indent=None)}")
        
        
        state_dict = pl_module.state_dict()
        filename = f"client_{self.idx}.pth"
        path = os.path.join(self.save_dir, filename)

        # Save the model state dictionary
        torch.save(state_dict, path)