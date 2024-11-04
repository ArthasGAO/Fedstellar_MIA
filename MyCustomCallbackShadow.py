import json
import torch
from lightning import Callback
import os

class MyCustomCheckpointShadow(Callback):
    def __init__(self, save_dir, epochs_of_interest):
        super().__init__()
        self.save_dir = save_dir
        self.epoch_set = set(epochs_of_interest)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch in self.epoch_set:
            
            state_dict = pl_module.state_dict()
            
            os.makedirs(self.save_dir, exist_ok=True)
            filename = f"model-epoch={epoch}.pth"
            path = os.path.join(self.save_dir, filename)

            # Save the model state dictionary
            torch.save(state_dict, path)