import torch

def init():
    global PARAM
    PARAM = {
        'model_data_name': 'MNIST',
        'train_data_name': 'MNIST',
        'test_data_name': 'MNIST',
        'model_dir': 'mnist',
        'model_name': 'iter_shuffle_codec',
        'resume_TAG': '',
        'special_TAG': '',
        'optimizer_name': 'Adam',
        'scheduler_name': 'ReduceLROnPlateau',
        'lr': 1e-3,
        'milestones': [10,20,50,100],
        'threshold': 0.5,
        'threshold_mode': 'abs',
        'factor': 0.5,
        'normalize': False,
        'batch_size': [50,500],
        'num_workers': 0,
        'data_size': 0,
        'device': 'cuda',
        'activation': 'tanh',
        'max_num_epochs': 200,
        'save_mode': 0,
        'world_size': 1,
        'train_metric_names': ['psnr','acc'],
        'test_metric_names': ['psnr','acc'],
        'topk': 1,
        'init_seed': 0,
        'num_Experiments': 1,
        'tuning_param': {'compression': 1, 'classification': 0},
        'loss_mode': {'compression':'mae','classification':'ce'},
        'num_level': 2,
        'code_size': 8,
        'num_iter': [16,8],
        'num_node': {'E':2,'D':1},
        'byclass': True,
        'resume_mode': 0
    }