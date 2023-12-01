data_config = {

    'root_dir': '/mnt/Jender/data',
    'dataset': 'ActivityNet',
    'batch_size': 16,
    'n_segment': 8,
    'sample_mod': 'sparing',
    'modality': 'RGB',
    'num_workers': 20,
    'pin_mem': True

}

model_config = {

    'num_frames': 8,
    'in_channels': 3,
    'num_classes': 200,
    'drop': 0.5,
    'pretrained': True,
    'pretrain_path': None,
    'fusion': 'avg',
    'base_model': 'resNet50'

}

train_config = {

    'epochs': 60,
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'top_k': (1, 5),
    'milestones': [20, 30, 40, 50],
    'manner': 'begin',
    'check_path': '/mnt/Jender/Exp/Ours/mods/ckp.pth',
    'save_path': '/mnt/Jender/Exp/Ours/mods/TEA-A200.pth',
    'log_file': '/mnt/Jender/Exp/Ours/mods/log.txt',
    'grad_clip': 20,
    'eval_freq': 1,
    'gpus': 1

}


