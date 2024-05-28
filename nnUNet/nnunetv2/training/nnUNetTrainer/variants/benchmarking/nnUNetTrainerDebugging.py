import torch
from batchgenerators.utilities.file_and_folder_operations import save_json, join, isfile, load_json

from nnunetv2.training.nnUNetTrainer.customTrainersULS import nnUNetTrainer_ULS_500_Robust
from torch import distributed as dist


class nnUNetTrainerDebugging(nnUNetTrainer_ULS_500_Robust):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        assert self.fold == 0, "It makes absolutely no sense to specify a certain fold. Stick with 0 so that we can parse the results."
        self.disable_checkpointing = True
        self.num_epochs = 1
        # assert torch.cuda.is_available(), "This only works on GPU"
        self.crashed_with_runtime_error = False

    def perform_actual_validation(self, save_probabilities: bool = False):
        pass

    def save_checkpoint(self, filename: str) -> None:
        # do not trust people to remember that self.disable_checkpointing must be True for this trainer
        pass

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()

    def on_train_end(self):
        super().on_train_end()

        if not self.is_ddp or self.local_rank == 0:
            torch_version = torch.__version__
            cudnn_version = torch.backends.cudnn.version()
            gpu_name = torch.cuda.get_device_name()
            if self.crashed_with_runtime_error:
                fastest_epoch = 'Not enough VRAM!'
            else:
                epoch_times = [i - j for i, j in zip(self.logger.my_fantastic_logging['epoch_end_timestamps'],
                                                     self.logger.my_fantastic_logging['epoch_start_timestamps'])]
                fastest_epoch = min(epoch_times)

            if self.is_ddp:
                num_gpus = dist.get_world_size()
            else:
                num_gpus = 1

            benchmark_result_file = join(self.output_folder, 'benchmark_result.json')
            if isfile(benchmark_result_file):
                old_results = load_json(benchmark_result_file)
            else:
                old_results = {}
            # generate some unique key
            my_key = f"{cudnn_version}__{torch_version.replace(' ', '')}__{gpu_name.replace(' ', '')}__gpus_{num_gpus}"
            old_results[my_key] = {
                'torch_version': torch_version,
                'cudnn_version': cudnn_version,
                'gpu_name': gpu_name,
                'fastest_epoch': fastest_epoch,
                'num_gpus': num_gpus,
            }
            save_json(old_results,
                      join(self.output_folder, 'benchmark_result.json'))
