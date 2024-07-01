from utils.utils import load_class


def make_task(encoder, task_cfg, encoder_config, dataset_config, train_config):
    task_class = load_class(task_cfg['task'])
    head_class = load_class(task_cfg['head'])
    
    head = head_class(encoder=encoder, **task_cfg['head_args'])  
    
    device = next(encoder.parameters()).device
    head.to(device)

    img_size = task_cfg['img_size']
    losses = task_cfg['losses']

    task = task_class(head, img_size, losses, encoder_config, dataset_config, train_config)

    return task
    
    
