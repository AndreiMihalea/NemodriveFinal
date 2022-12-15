import torch
import torch.nn as nn


def get_state_dict_on_cpu(obj):
    cpu_device = torch.device('cpu')
    state_dict = obj.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(cpu_device)
    return state_dict


def save_ckpt(ckpt_name, models, optimizers, schedulers, rlosses, best_scores, n_iter):
    ckpt_dict = {'n_iter': n_iter}
    for prefix, model in models:
        ckpt_dict[prefix] = get_state_dict_on_cpu(model)

    for prefix, optimizer in optimizers:
        ckpt_dict[prefix] = optimizer.state_dict()

    for prefix, scheduler in schedulers:
        ckpt_dict[prefix] = scheduler

    for prefix, rloss in rlosses:
        ckpt_dict[prefix] = rloss
    
    for prefix, best_score in best_scores:
        ckpt_dict[prefix] = best_scores

    torch.save(ckpt_dict, ckpt_name)


def load_ckpt(ckpt_name, models, optimizers=None, schedulers=None, rlosses=None, best_scores=None):
    ckpt_dict = torch.load(ckpt_name)
    for prefix, model in models:
        assert isinstance(model, nn.Module)
        model.load_state_dict(ckpt_dict[prefix], strict=False)

    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])
    
    if schedulers is not None:
        for prefix, scheduler in schedulers:
            scheduler = ckpt_dict[prefix] 

    if rlosses is not None:
        for prefix, rloss in rlosses:
            rloss = ckpt_dict[prefix]

    if best_scores is not None:
        for prefix, best_score in best_scores:
            best_score = ckpt_dict[prefix]
            
    return ckpt_dict['n_iter']
