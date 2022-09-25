import torch
from sys import platform


class TrainGlobalConfig:
    continue_gdrive = None  # this is for loading pretrained weights from a gdrive folder

    device = torch.device("cuda")
    num_folds = 6

    num_workers = 0 if platform == "win32" else 8

    n_epochs = 25
    if continue_gdrive is not None: n_epochs = 1

    lr = 0.0001
    batch_size = 3

    n_crops_in_train = 5
    n_crops_in_val = 7
    cropx = 200  # time
    cropy = 200  # mels

    verbose = True
    verbose_step = 1
    folder = "./model/"
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss

    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=3,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=0.00001,
        eps=1e-08
    )
