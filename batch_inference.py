import os
import logging
import codeLib
import ssg
import ssg.config as config
from ssg.checkpoints import CheckpointIO
from ssg.dataset.custom_dataloader_3DSSG import Custom3DSSGDataset
import cProfile
import matplotlib
import torch
import torch_geometric
import networkx as nx
import torch
from torch import nn
import codeLib.utils.string_numpy as snp
import json
import torch

def main():
    print(f"Start")
    cfg = ssg.Parse()
    out_dir = os.path.join(cfg['training']['out_dir'], cfg.name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # set random seed
    codeLib.utils.util.set_random_seed(cfg.SEED)

    cfg.data.load_cache = False

    n_workers = cfg['training']['data_workers']

    print(f"After loading config")
    mode = cfg.MODE
    custom_dataset = Custom3DSSGDataset(cfg, mode)

    class_names = custom_dataset.class_names
    relationship_names = custom_dataset.relationship_names
    num_class_names = len(class_names)
    num_relationship_names = len(relationship_names)
    '''load pretrained model'''
    model = config.get_model(
         cfg, num_obj_cls=num_class_names, num_rel_cls=num_relationship_names)
    #if not os.path.ex
    checkpoint_io = CheckpointIO(out_dir, model=model)
    load_dict = checkpoint_io.load('model_best.pt', device=cfg.DEVICE) # dict_keys(['epoch_it', 'it', 'loss_val_best', 'patient', 'optimizer', 'scheduler', 'smoother'])
    it = load_dict.get('it', -1)
    test_loader = torch_geometric.loader.DataLoader(
        custom_dataset, batch_size=1, num_workers=n_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=False
    )
    '''Inference'''
    model.eval()
    obj_logits = None 
    rel_logits = None
    obj_cls_predictions = None
    rel_cls_predictions = None
    pth_experiment = "./experiments/custom_dataset/custom_3DSSG_full_l160"
    with torch.no_grad():
        softmax = nn.Softmax(dim=1)
        for sample in test_loader:
            print(f"sample: {sample}")
            sample = sample.to(cfg.DEVICE)
            obj_logits, rel_logits = model(sample)
            obj_cls_predictions = torch.softmax(obj_logits, dim=1)
            rel_cls_predictions = torch.sigmoid(rel_logits)
            rel_cls_predictions = rel_logits > 0.5
            # torch.save(sample, os.path.join(pth_experiment, "sample_gt.pt")) # sample.pt
            # torch.save(obj_logits, os.path.join(pth_experiment, "obj_logits_gt.pt")) # obj_logits.pt
            # torch.save(rel_logits, os.path.join(pth_experiment, "rel_logits_gt.pt")) # rel_logits.pt
            # torch.save(obj_cls_predictions, os.path.join(pth_experiment, "obj_cls_predictions_gt.pt")) # obj_cls_predictions.pt
            # torch.save(rel_cls_predictions, os.path.join(pth_experiment, "rel_cls_predictions_gt.pt")) # rel_cls_predictions.pt

if __name__ == '__main__':
    main()