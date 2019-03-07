import torch

def init():
    global PARAM
    PARAM = {
        'model_data_name': 'SVHN',
        'train_data_name': 'SVHN',
        'test_data_name': 'SVHN',
        'model_dir': 'mnist',
        'model_name': 'rescodec',
        'resume_TAG': '',
        'special_TAG': '',
        'optimizer_name': 'Adam',
        'scheduler_name': 'ReduceLROnPlateau',
        'lr': 1e-3,
        'milestones': [10,20,50,100],
        'threshold': 0.1,
        'threshold_mode': 'abs',
        'factor': 0.5,
        'normalize': False,
        'batch_size': [100,500],
        'num_workers': 0,
        'data_size': 0,
        'device': 'cuda',
        'activation': 'relu',
        'max_num_epochs': 200,
        'save_mode': 0,
        'world_size': 1,
        'train_metric_names': ['psnr','acc'],
        'test_metric_names': ['psnr','acc'],
        'topk': 1,
        'init_seed': 0,
        'num_Experiments': 1,
        'tuning_param': {'compression': 0, 'classification': 1},
        'loss_mode': {'compression':'mae','classification':'ce'},
        'num_level': 2,
        'code_size': 128,
        'num_iter': 16,
        'patch_shape': (32,32),
        'step': [1.0,1.0],
        'jump_rate': 2,
        'train_num_node': 1,
        'test_num_node': 1,
        'resume_mode': 0
    }