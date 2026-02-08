# Original copyright:
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Github repo: https://github.com/google-research/semivl/blob/main/utils/train_utils.py

import numpy as np
import torch
from typing import Callable, Dict, List, Optional, Sequence, Union
from mmengine.optim import (OptimWrapper, OptimWrapperDict, _ParamScheduler,
                            build_optim_wrapper)
ParamSchedulerType = Union[List[_ParamScheduler], Dict[str,
                                                       List[_ParamScheduler]]]

import copy
from mmengine.registry import PARAM_SCHEDULERS

def cutmix_img_(img, img_mix, cutmix_box):
    img[cutmix_box.unsqueeze(1).expand(img.shape) == 1] = \
        img_mix[cutmix_box.unsqueeze(1).expand(img.shape) == 1]
  

def cutmix_mask(mask, mask_mix, cutmix_box):
    cutmixed = mask.clone()
    cutmixed[cutmix_box == 1] = mask_mix[cutmix_box == 1]
    return cutmixed


def softmix_img_(img, img_mix, cutmix_box, beta):
    """
    img:        原图像，形状为 (B, C, H, W)
    img_mix:    用于粘贴的图像，形状为 (B, C, H, W)
    cutmix_box: 粘贴区域掩码，形状为 (B, H, W)，1 表示粘贴区域
    beta:       透明度系数（软粘贴强度），范围 [0, 1]
    """
    # 扩展 cutmix_box 为 (B, C, H, W)
    return beta * img_mix + (1 - beta) * img


def softmix_pre_(pre, pre_mix, cutmix_box, beta):
    """
    mask:       原始标签，形状为 (B, H, W)
    mask_mix:   用于粘贴的标签，形状为 (B, H, W)
    cutmix_box: 粘贴区域掩码，形状为 (B, H, W)，1 表示粘贴区域
    beta:       透明度系数，范围 [0, 1]
    """
    return beta * pre_mix + (1 - beta) * pre

def confidence_weighted_loss(loss, conf_map, ignore_mask, conf_thresh=0.95, conf_mode='pixelwise'):
    assert loss.dim() == 3
    assert conf_map.dim() == 3
    assert ignore_mask.dim() == 3
    valid_mask = (ignore_mask != 255)
    # print("valid_mask", valid_mask)
    sum_pixels = dict(dim=(1, 2), keepdim=True)
    if conf_mode == 'pixelwise':
        loss = loss * ((conf_map >= conf_thresh) & valid_mask)
        loss = loss.sum() / valid_mask.sum().item()
    elif conf_mode == 'pixelratio':
        ratio_high_conf = ((conf_map >= conf_thresh) & valid_mask).sum(**sum_pixels) / valid_mask.sum(**sum_pixels)
        loss = loss * ratio_high_conf
        loss = loss.sum() / valid_mask.sum().item()
    elif conf_mode == 'pixelavg':
        avg_conf = (conf_map * valid_mask).sum(**sum_pixels) / valid_mask.sum(**sum_pixels)
        loss = loss.sum() * avg_conf
        loss = loss.sum() / valid_mask.sum().item()
    else:
        raise ValueError(conf_mode)
    return loss


# def confidence_weighted_loss(loss, em_map, ignore_mask, em_threshold, conf_mode='pixelwise'):
#     assert loss.dim() == 3
#     assert em_map.dim() == 3
#     assert ignore_mask.dim() == 3
#     valid_mask = (ignore_mask != 255)
#     # print("valid_mask", valid_mask)
#     sum_pixels = dict(dim=(1, 2), keepdim=True)
#     if conf_mode == 'pixelwise':
#         loss = loss * ((em_map <= em_threshold) & valid_mask)
#         loss = loss.sum() / valid_mask.sum().item()
#     elif conf_mode == 'pixelratio':
#         ratio_high_conf = ((conf_map >= conf_thresh) & valid_mask).sum(**sum_pixels) / valid_mask.sum(**sum_pixels)
#         loss = loss * ratio_high_conf
#         loss = loss.sum() / valid_mask.sum().item()
#     elif conf_mode == 'pixelavg':
#         avg_conf = (conf_map * valid_mask).sum(**sum_pixels) / valid_mask.sum(**sum_pixels)
#         loss = loss.sum() * avg_conf
#         loss = loss.sum() / valid_mask.sum().item()
#     else:
#         raise ValueError(conf_mode)
#     return loss



def build_param_scheduler(optim_wrapper, 
                          scheduler: Union[_ParamScheduler, Dict, List],
                          train_dataloader) -> ParamSchedulerType:
    """Build parameter schedulers.

    ``build_param_scheduler`` should be called after
    ``build_optim_wrapper`` because the building logic will change
    according to the number of optimizers built by the runner.
    The cases are as below:

    - Single optimizer: When only one optimizer is built and used in the
        runner, ``build_param_scheduler`` will return a list of
        parameter schedulers.
    - Multiple optimizers: When two or more optimizers are built and used
        in runner, ``build_param_scheduler`` will return a dict containing
        the same keys with multiple optimizers and each value is a list of
        parameter schedulers. Note that, if you want different optimizers to
        use different parameter schedulers to update optimizer's
        hyper-parameters, the input parameter ``scheduler`` also needs to be
        a dict and its key are consistent with multiple optimizers.
        Otherwise, the same parameter schedulers will be used to update
        optimizer's hyper-parameters.

    Args:
        scheduler (_ParamScheduler or dict or list): A Param Scheduler
            object or a dict or list of dict to build parameter schedulers.

    Examples:
        >>> # build one scheduler
        >>> optim_cfg = dict(dict(type='SGD', lr=0.01))
        >>> runner.optim_wrapper = runner.build_optim_wrapper(
        >>>     optim_cfg)
        >>> scheduler_cfg = dict(type='MultiStepLR', milestones=[1, 2])
        >>> schedulers = runner.build_param_scheduler(scheduler_cfg)
        >>> schedulers
        [<mmengine.optim.scheduler.lr_scheduler.MultiStepLR at 0x7f70f6966290>]  # noqa: E501

        >>> # build multiple schedulers
        >>> scheduler_cfg = [
        ...    dict(type='MultiStepLR', milestones=[1, 2]),
        ...    dict(type='StepLR', step_size=1)
        ... ]
        >>> schedulers = runner.build_param_scheduler(scheduler_cfg)
        >>> schedulers
        [<mmengine.optim.scheduler.lr_scheduler.MultiStepLR at 0x7f70f60dd3d0>,  # noqa: E501
        <mmengine.optim.scheduler.lr_scheduler.StepLR at 0x7f70f6eb6150>]

    Above examples only provide the case of one optimizer and one scheduler
    or multiple schedulers. If you want to know how to set parameter
    scheduler when using multiple optimizers, you can find more examples
    `optimizer-docs`_.

    Returns:
        list[_ParamScheduler] or dict[str, list[_ParamScheduler]]: List of
        parameter schedulers or a dictionary contains list of parameter
        schedulers build from ``scheduler``.

    .. _optimizer-docs:
        https://mmengine.readthedocs.io/en/latest/tutorials/optim_wrapper.html
    """
    param_schedulers: ParamSchedulerType

    # param_schedulers = dict()
    # for name, optimizer in optim_wrapper.items():
    #     if isinstance(scheduler, dict) and 'type' not in scheduler:
    #         # scheduler is a dict and each item is a ParamScheduler
    #         # object or a config to build ParamScheduler objects
    #         param_schedulers[name] = _build_param_scheduler(
    #             scheduler[name], optimizer, train_dataloader)
    #     else:
    #         param_schedulers[name] = _build_param_scheduler(
    #             scheduler, optimizer, train_dataloader)

    # return param_schedulers
    if not isinstance(optim_wrapper, OptimWrapperDict):
            # Since `OptimWrapperDict` inherits from `OptimWrapper`,
            # `isinstance(self.optim_wrapper, OptimWrapper)` cannot tell
            # whether `self.optim_wrapper` is an `OptimizerWrapper` or
            # `OptimWrapperDict` instance. Therefore, here we simply check
            # self.optim_wrapper is not an `OptimWrapperDict` instance and
            # then assert it is an OptimWrapper instance.
        assert isinstance(optim_wrapper, OptimWrapper), (
            '`build_optimizer` should be called before'
            '`build_param_scheduler` because the latter depends '
            'on the former')
        param_schedulers = _build_param_scheduler(
            scheduler, optim_wrapper, train_dataloader)  # type: ignore
        return param_schedulers
    else:
        param_schedulers = dict()
        for name, optimizer in optim_wrapper.items():
            if isinstance(scheduler, dict) and 'type' not in scheduler:
                # scheduler is a dict and each item is a ParamScheduler
                # object or a config to build ParamScheduler objects
                param_schedulers[name] = _build_param_scheduler(
                    scheduler[name], optimizer)
            else:
                param_schedulers[name] = _build_param_scheduler(
                    scheduler, optimizer)

    return param_schedulers



def scheduler_after_train_iter(param_schedulers) -> None:
    """Call step function for each scheduler after each training iteration.

    Args:
        runner (Runner): The runner of the training process.
        batch_idx (int): The index of the current batch in the train loop.
        data_batch (dict or tuple or list, optional): Data from dataloader.
            In order to keep this interface consistent with other hooks,
            we keep ``data_batch`` here.
        outputs (dict, optional): Outputs from model.
            In order to keep this interface consistent with other hooks, we
            keep ``data_batch`` here.
    """

    if param_schedulers is None:
        return

    def step(param_schedulers):
        assert isinstance(param_schedulers, list)
        for scheduler in param_schedulers:
            if not scheduler.by_epoch:
                scheduler.step()

    if isinstance(param_schedulers, list):
        step(param_schedulers)
    elif isinstance(param_schedulers, dict):
        for param_schedulers in param_schedulers.values():
            step(param_schedulers)
    else:
        raise TypeError(
            'runner.param_schedulers should be list of ParamScheduler or '
            'a dict containing list of ParamScheduler, '
            f'but got {param_schedulers}')


def _build_param_scheduler(
        scheduler: Union[_ParamScheduler, Dict, List],
        optim_wrapper: OptimWrapper,
        train_dataloader) -> List[_ParamScheduler]:
    """Build parameter schedulers for a single optimizer.

    Args:
        scheduler (_ParamScheduler or dict or list): A Param Scheduler
            object or a dict or list of dict to build parameter schedulers.
        optim_wrapper (OptimWrapper): An optimizer wrapper object is
            passed to construct ParamScheduler object.

    Returns:
        list[_ParamScheduler]: List of parameter schedulers build from
        ``scheduler``.

    Note:
        If the train loop is built, when building parameter schedulers,
        it supports setting the max epochs/iters as the default ``end``
        of schedulers, and supports converting epoch-based schedulers
        to iter-based according to the ``convert_to_iter_based`` key.
    """
    if not isinstance(scheduler, Sequence):
        schedulers = [scheduler]
    else:
        schedulers = scheduler

    param_schedulers = []
    for scheduler in schedulers:
        if isinstance(scheduler, _ParamScheduler):
            param_schedulers.append(scheduler)
        elif isinstance(scheduler, dict):
            _scheduler = copy.deepcopy(scheduler)

            param_schedulers.append(
                PARAM_SCHEDULERS.build(
                    _scheduler,
                    default_args=dict(
                        optimizer=optim_wrapper,
                        epoch_length=len(train_dataloader))))
                        # epoch_length=2000)))

        else:
            raise TypeError(
                'scheduler should be a _ParamScheduler object or dict, '
                f'but got {scheduler}')
    return param_schedulers


class DictAverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avgs = {}
        self.sums = {}
        self.counts = {}

    def update(self, vals):
        for k, v in vals.items():
            if torch.is_tensor(v):
                v = v.detach()
            if k not in self.sums:
                self.sums[k] = 0
                self.counts[k] = 0
            self.sums[k] += v
            self.counts[k] += 1
            self.avgs[k] = torch.true_divide(self.sums[k], self.counts[k])

    def __str__(self):
        s = []
        for k, v in self.avgs.items():
            s.append(f'{k}: {v:.3f}')
        return ', '.join(s)
    

def generate_lambda_schedule(epochs, total_epochs, warmup_epochs):
    if epochs < warmup_epochs:
        lambda_values = (epochs / warmup_epochs)
    else:
        lambda_values = (1 - (epochs - warmup_epochs) / (total_epochs - warmup_epochs))
    return lambda_values



class DictAverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avgs = {}
        self.sums = {}
        self.counts = {}

    def update(self, vals):
        for k, v in vals.items():
            if torch.is_tensor(v):
                v = v.detach()
            if k not in self.sums:
                self.sums[k] = 0
                self.counts[k] = 0
            self.sums[k] += v
            self.counts[k] += 1
            self.avgs[k] = torch.true_divide(self.sums[k], self.counts[k])

    def __str__(self):
        s = []
        for k, v in self.avgs.items():
            s.append(f'{k}: {v:.3f}')
        return ', '.join(s)
    

def generate_lambda_schedule(epochs, total_epochs, warmup_epochs):
    if epochs < warmup_epochs:
        lambda_values = (epochs / warmup_epochs)
    else:
        lambda_values = (1 - (epochs - warmup_epochs) / (total_epochs - warmup_epochs))
    return lambda_values
