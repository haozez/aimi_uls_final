import torch
import numpy as np
import gc

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import \
    DC_and_topk_loss, \
    DC_and_focal_loss, \
    DC_and_BCE_robust_loss, \
    DC_and_CE_robust_loss, \
    DC_CE_Axis_loss, \
    DC_and_topk_robust_loss

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from torch import autocast
from nnunetv2.utilities.helpers import dummy_context

class nnUNetTrainer_ULS_3k(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 3000

class nnUNetTrainer_ULS_3k_HalfLR(nnUNetTrainer_ULS_3k):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 5e-3

class nnUNetTrainer_ULS_3k_QuarterLR(nnUNetTrainer_ULS_3k):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 2.5e-3

class nnUNetTrainer_ULS_2k(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 2000

class nnUNetTrainer_ULS_2k_HalfLR(nnUNetTrainer_ULS_2k):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 5e-3

class nnUNetTrainer_ULS_2k_QuarterLR(nnUNetTrainer_ULS_2k):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 2.5e-3

class nnUNetTrainer_ULS_1k_HalfLR(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 5e-3

class nnUNetTrainer_ULS_1k_QuarterLR(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 2.5e-3

class nnUNetTrainer_ULS_500_HalfLR(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 5e-3
        self.num_epochs = 500

class nnUNetTrainer_ULS_500(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500

class nnUNetTrainer_ULS_500_QuarterLR(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 2.5e-3
        self.num_epochs = 500

class nnUNetTrainer_ULS_500_Robust(nnUNetTrainer_ULS_500):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)    

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_robust_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_robust_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
    
    def train_step(self, batch: dict) -> dict:
        # Modified to forward pass on inputs rotated by 180 degrees 
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True) 
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(torch.concatenate((data, torch.rot90(data, k=2, dims=[3, 4])), axis=0))
            data.detach().cpu()
            del data
            gc.collect()
            torch.cuda.empty_cache()
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}
    
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # Concatenate original input with version rotated 180 degrees
            output = self.network(torch.concatenate((data, torch.rot90(data, k=2, dims=[3, 4])), axis=0))
            data.detach().cpu()
            del data
            gc.collect()
            torch.cuda.empty_cache()
            l = self.loss(output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}    

class nnUNetTrainer_ULS_DCTopKLoss_Robust(nnUNetTrainer_ULS_500_Robust):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # Extending previous robust trainer avoids re-defining the custom
        # training and validation functionality
        self.initial_lr = 1e-2
        self.num_epochs = 500

    def _build_loss(self):
            if self.label_manager.has_regions:
                loss = DC_and_topk_robust_loss({'batch_dice': self.configuration_manager.batch_dice,
                                        'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp}, {},
                                        ignore_label=self.label_manager.ignore_label is not None)
            else:
                loss = DC_and_topk_robust_loss({'batch_dice': self.configuration_manager.batch_dice,
                                    'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                    ignore_label=self.label_manager.ignore_label)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss

            if self.enable_deep_supervision:
                deep_supervision_scales = self._get_deep_supervision_scales()
                weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
                if self.is_ddp and not self._do_i_compile():
                    # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                    # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                    # Anywho, the simple fix is to set a very low weight to this.
                    weights[-1] = 1e-6
                else:
                    weights[-1] = 0

                # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
                weights = weights / weights.sum()
                # now wrap the loss
                loss = DeepSupervisionWrapper(loss, weights)
            return loss

class nnUNetTrainer_ULS_DCTopKLoss(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-2
        self.num_epochs = 500

    def _build_loss(self):
            if self.label_manager.has_regions:
                loss = DC_and_topk_loss({'batch_dice': self.configuration_manager.batch_dice,
                                        'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp}, {},
                                        ignore_label=self.label_manager.ignore_label is not None)
            else:
                loss = DC_and_topk_loss({'batch_dice': self.configuration_manager.batch_dice,
                                    'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                    ignore_label=self.label_manager.ignore_label)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss

            if self.enable_deep_supervision:
                deep_supervision_scales = self._get_deep_supervision_scales()
                weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
                if self.is_ddp and not self._do_i_compile():
                    # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                    # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                    # Anywho, the simple fix is to set a very low weight to this.
                    weights[-1] = 1e-6
                else:
                    weights[-1] = 0

                # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
                weights = weights / weights.sum()
                # now wrap the loss
                loss = DeepSupervisionWrapper(loss, weights)
            return loss

class nnUNetTrainer_ULS_DCFocalLoss(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-2
        self.num_epochs = 500

    def _build_loss(self):
            if self.label_manager.has_regions:
                loss = DC_and_focal_loss({'batch_dice': self.configuration_manager.batch_dice,
                                        'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp}, {},
                                        ignore_label=self.label_manager.ignore_label is not None)
            else:
                loss = DC_and_focal_loss({'batch_dice': self.configuration_manager.batch_dice,
                                    'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                    ignore_label=self.label_manager.ignore_label)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss

            if self.enable_deep_supervision:
                deep_supervision_scales = self._get_deep_supervision_scales()
                weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
                if self.is_ddp and not self._do_i_compile():
                    # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                    # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                    # Anywho, the simple fix is to set a very low weight to this.
                    weights[-1] = 1e-6
                else:
                    weights[-1] = 0

                # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
                weights = weights / weights.sum()
                # now wrap the loss
                loss = DeepSupervisionWrapper(loss, weights)
            return loss

class nnUNetTrainer_ULS_DCFocalLoss_TestOnly(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-2
        self.num_epochs = 10

    def _build_loss(self):
            if self.label_manager.has_regions:
                loss = DC_and_focal_loss({'batch_dice': self.configuration_manager.batch_dice,
                                        'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp}, {},
                                        ignore_label=self.label_manager.ignore_label is not None)
            else:
                loss = DC_and_focal_loss({'batch_dice': self.configuration_manager.batch_dice,
                                    'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                    ignore_label=self.label_manager.ignore_label)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss

            if self.enable_deep_supervision:
                deep_supervision_scales = self._get_deep_supervision_scales()
                weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
                if self.is_ddp and not self._do_i_compile():
                    # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                    # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                    # Anywho, the simple fix is to set a very low weight to this.
                    weights[-1] = 1e-6
                else:
                    weights[-1] = 0

                # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
                weights = weights / weights.sum()
                # now wrap the loss
                loss = DeepSupervisionWrapper(loss, weights)
            return loss

class nnUNetTrainer_ULS_DCCEAxisLoss(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-2
        self.num_epochs = 500

    def _build_loss(self):
            if self.label_manager.has_regions:
                loss = DC_CE_Axis_loss({}, {'batch_dice': self.configuration_manager.batch_dice,
                                        'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                        use_ignore_label=self.label_manager.ignore_label is not None)
            else:
                loss = DC_CE_Axis_loss({}, {'batch_dice': self.configuration_manager.batch_dice,
                                    'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                                    use_ignore_label=self.label_manager.ignore_label)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss

            if self.enable_deep_supervision:
                deep_supervision_scales = self._get_deep_supervision_scales()
                weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
                if self.is_ddp and not self._do_i_compile():
                    # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                    # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                    # Anywho, the simple fix is to set a very low weight to this.
                    weights[-1] = 1e-6
                else:
                    weights[-1] = 0

                # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
                weights = weights / weights.sum()
                # now wrap the loss
                loss = DeepSupervisionWrapper(loss, weights)
            return loss