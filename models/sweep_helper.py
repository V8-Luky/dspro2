import torch
import torch.nn as nn

from enum import Enum


TYPE = "type"


class OptimizerType:
    ADAM = "adam"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"


OPTIMIZER = "optimizer"
LEARNING_RATE = "learning_rate"
FINETUNE_LEARNING_RATE = "finetune_learning_rate"
WEIGHT_DECAY = "weight_decay"
MOMENTUM = "momentum"


def get_optimizer(optimizer_params: dict, model: nn.Module):
    optimizer = optimizer_params[TYPE]
    learning_rate = optimizer_params[LEARNING_RATE]
    weight_decay = optimizer_params[WEIGHT_DECAY]

    if optimizer == OptimizerType.ADAM:
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == OptimizerType.ADAMW:
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == OptimizerType.RMSPROP:
        momentum = optimizer_params[MOMENTUM]
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

def get_optimizer_with_finetune_group(optimizer_params: dict, model: nn.Module) -> torch.optim.Optimizer:
    optimizer = optimizer_params[TYPE]
    learning_rate = optimizer_params[LEARNING_RATE]
    finetune_learning_rate = optimizer_params[FINETUNE_LEARNING_RATE]    
    weight_decay = optimizer_params[WEIGHT_DECAY]

    parameters = [
        {"params": model.get_finetune_params(), "lr": finetune_learning_rate},
        {"params": model.get_main_params(), "lr": learning_rate},
    ]

    if optimizer == OptimizerType.ADAM:
        return torch.optim.Adam(parameters, weight_decay=weight_decay)
    elif optimizer == OptimizerType.ADAMW:
        return torch.optim.AdamW(parameters, weight_decay=weight_decay)
    elif optimizer == OptimizerType.RMSPROP:
        momentum = optimizer_params[MOMENTUM]
        return torch.optim.RMSprop(parameters, weight_decay=weight_decay, momentum=momentum)


class LearningRateSchedulerType:
    NONE = "none"
    STEP = "step"
    EXPONENTIAL = "exponential"
    CONSTANT = "constant"


LEARNING_RATE_SCHEDULER = "learning_rate_scheduler"
GAMMA = "gamma"
STEP_SIZE = "step_size"
FACTOR = "factor"


def get_learning_rate_scheduler(learning_rate_scheduler_params: dict, optimizer: torch.optim.Optimizer):
    learning_rate_scheduler = learning_rate_scheduler_params[TYPE]
    if learning_rate_scheduler == LearningRateSchedulerType.NONE:
        return None

    if learning_rate_scheduler == LearningRateSchedulerType.STEP:
        step_size = learning_rate_scheduler_params[STEP_SIZE]
        gamma = learning_rate_scheduler_params[GAMMA]
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif learning_rate_scheduler == LearningRateSchedulerType.EXPONENTIAL:
        gamma = learning_rate_scheduler_params[GAMMA]
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif learning_rate_scheduler == LearningRateSchedulerType.CONSTANT:
        factor = learning_rate_scheduler_params[FACTOR]
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=factor)
