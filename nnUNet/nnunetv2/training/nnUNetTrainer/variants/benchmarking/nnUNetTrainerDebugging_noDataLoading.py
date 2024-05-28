import torch

from nnunetv2.training.nnUNetTrainer.variants.benchmarking.nnUNetTrainerDebugging import (
    nnUNetTrainerDebugging,
)
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.helpers import dummy_context
from torch import autocast

class nnUNetTrainerDebugging_noDataLoading(nnUNetTrainerDebugging):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self._set_batch_size_and_oversample()
        num_input_channels = determine_num_input_channels(
            self.plans_manager, self.configuration_manager, self.dataset_json
        )
        patch_size = self.configuration_manager.patch_size
        dummy_data = torch.rand((self.batch_size, num_input_channels, *patch_size), device=self.device)
        if self.enable_deep_supervision:
            dummy_target = [
                torch.round(
                    torch.rand((self.batch_size, 1, *[int(i * j) for i, j in zip(patch_size, k)]), device=self.device)
                    * max(self.label_manager.all_labels)
                )
                for k in self._get_deep_supervision_scales()
            ]
        else:
            raise NotImplementedError("This trainer does not support deep supervision")
        self.dummy_batch = {"data": dummy_data, "target": dummy_target}

    def get_dataloaders(self):
        return None, None

    def run_training(self):
        self.on_train_start()
        self.on_epoch_start()
        self.on_train_epoch_start()

        # Just perform a single forward pass
        out = self.train_step(self.dummy_batch)

        self.on_train_epoch_end([out])
        self.on_epoch_end()
        self.on_train_end()
            

