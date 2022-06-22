from pathlib import Path

from sklearn.metrics import log_loss
thispath = Path.cwd().resolve()
import sys; sys.path.insert(0, str(thispath.parent))

import logging
import random
import time
import torch
import torchvision
import yaml

import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import FasterRCNN, RetinaNet
from torchvision.models.detection.rpn import AnchorGenerator

import deep_learning.dl_utils as dl_utils
from deep_learning.dataset.dataset import INBreast_Dataset_pytorch
from deep_learning.classification_models.base_classifier import CNNClasssifier
from deep_learning.detection_models.vision_utils.coco_utils import get_coco_api_from_dataset
from deep_learning.detection_models.vision_utils.engine import train_one_epoch, evaluation

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def identity_function(arg):
    return arg


def train_model(datasets, dataloaders, model, optimizer, scheduler, cfg):

    # guarantee reproducibility
    since = time.time()
    random.seed(0)
    torch.manual_seed(1442)
    np.random.seed(0)

    # holders for best model
    best_metric = 0.0
    best_epoch = 0
    last_three_losses = []
    early_stopping_count = 0
    previous_mean_loss = 0
    best_metric_name = cfg['training']['best_metric']

    exp_path = Path.cwd().parent.parent/f'data/deepl_runs/{cfg["experiment_name"]}'
    exp_path.mkdir(exist_ok=True, parents=True)
    best_model_path = exp_path / f'{cfg["experiment_name"]}_{best_metric_name}.pt'
    chkpt_path = exp_path / f'{cfg["experiment_name"]}_chkpt.pt'
    logging.info(f'Storing experiment in: {exp_path}')

    if cfg['training']['resume_training']:
        checkpoint = torch.load(chkpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['epoch'] + 1
    else:
        init_epoch = 0

    # tensorboard loggs
    log_dir = exp_path/'tensorboard'
    log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=log_dir)

    coco_val = get_coco_api_from_dataset(dataloaders['val'].dataset)

    for epoch in range(init_epoch, cfg['training']['n_epochs']):
        # train for one epoch, printing every 10 iterations
        _, epoch_loss = train_one_epoch(
            model, optimizer, dataloaders['train'], device, epoch, cfg, tensorboard_writer=writer)

        # update the learning rate
        scheduler.step()

        # evaluate on the test dataset
        _, metrics = evaluation(model, dataloaders['val'], device, coco_val)

        # compute and log the metrics for the epoch
        last_three_losses.append(epoch_loss)
        if len(last_three_losses) > 3:
            last_three_losses = last_three_losses[1:]

        # print status
        message = \
            f'Train Loss: {epoch_loss:.4f}, AP_IoU_0.50_all: {metrics["AP_IoU_0.50_all"]:.4f}' \
            f'AR_IoU_0.50_0.95_all_mdets_100: {metrics["AR_IoU_0.50_0.95_all_mdets_100"]}'
        logging.info(message)
        writer.add_scalar("AP_IoU_0.50_all/val", metrics['AP_IoU_0.50_all'], epoch+1)
        writer.add_scalar(
            'AR_IoU_0.50_0.95_all_mdets_100/val',
            metrics['AR_IoU_0.50_0.95_all_mdets_100'], epoch+1)
        writer.add_scalar("Loss/train", epoch_loss, epoch+1)

        # save last and best checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'configuration': cfg,
            'loss': epoch_loss}, chkpt_path)

        if metrics[best_metric_name] > best_metric:
            best_metric = metrics[best_metric_name]
            best_epoch = epoch+1
            torch.save({
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
                'configuration': cfg
                }, best_model_path)

        if cfg['training']['early_stopping'] and (epoch != 0):
            diff = np.mean(last_three_losses) - previous_mean_loss
            if -diff < cfg['training']['early_stopping_args']['min_diff']:
                early_stopping_count += 1
            else:
                early_stopping_count = 0
        previous_mean_loss = np.mean(last_three_losses)

        if cfg['training']['early_stopping']:
            max_epochs = cfg['training']['early_stopping_args']['max_epoch']
            if early_stopping_count == max_epochs:
                msg = f'Early stopping after {max_epochs} epochs without' \
                    f' significant change in val metric'
                logging.info(msg)
                break
    logging.info(('-' * 10))

    time_elapsed = time.time() - since
    message = f'Training complete in {(time_elapsed // 60):.0f}m ' \
        f'{(time_elapsed % 60):.0f}s'
    logging.info(message)
    logging.info(
        f'Best val AP_IoU_0.50_all/val: {metrics["AP_IoU_0.50_all"]:4f},'
        f' AR_IoU_0.50_0.95_all: {metrics["AR_IoU_0.50_0.95_all_mdets_100"]:4f},'
        f' epoch {best_epoch}')

    # close the tensorboard session
    writer.flush()
    writer.close()

    # load best model weights before returning
    best_model = torch.load(best_model_path)
    model.load_state_dict(best_model['model_state_dict'])
    return model


def main():
    # read the configuration file
    config_path = str(thispath.parent.parent/'calc-det/deep_learning/detection_models/config.yml')
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    # for colab it needs to be specified additionally
    dataset_arguments = cfg['dataset']

    # use the configuration for the dataset
    dataset_arguments = cfg['dataset']
    dataset_arguments['patch_images_path'] = Path(dataset_arguments['patch_images_path'])
    datasets = {
        'train': INBreast_Dataset_pytorch(
            partitions=['train'], neg_to_pos_ratio=dataset_arguments['train_neg_to_pos_ratio'],
            **dataset_arguments),
        'val': INBreast_Dataset_pytorch(
            partitions=['validation'], neg_to_pos_ratio=None, **dataset_arguments)
    }

    # use the configuration for the dataloaders
    def collate_fn(batch):
        return tuple(zip(*batch))
    dataloaders = {
        'val': DataLoader(
            datasets['val'], batch_size=cfg['dataloaders']['val_batch_size'],
            num_workers=4, collate_fn=collate_fn, drop_last=False),
        'train': DataLoader(
            datasets['train'], batch_size=cfg['dataloaders']['train_batch_size'],
            shuffle=True, collate_fn=collate_fn, num_workers=4, drop_last=False)
    }

    # model settings
    if cfg['model']['checkpoint_path'] is not None:
        model_ckpt = torch.load(cfg['model']['checkpoint_path'])
        model = dl_utils.get_model_from_checkpoint(model_ckpt, cfg['model']['freeze_weights'])
    else:
        model = CNNClasssifier(
            activation=getattr(nn, cfg['model']['activation'])(),
            dropout=cfg['model']['dropout'],
            fc_dims=cfg['model']['fc_dims'],
            freeze_weights=cfg['model']['freeze_weights'],
            backbone=cfg['model']['backbone'],
            pretrained=cfg['model']['pretrained'],
        )
        model = model.model

    modules = list(model.children())[:-2]      # delete the last fc layers.
    last_submodule_childs = list(modules[-1][-1].children())
    for i in range(len(last_submodule_childs)-1, -1, -1):
        if hasattr(last_submodule_childs[i], 'out_channels'):
            out_channels = last_submodule_childs[i].out_channels
            break
    model_backbone = nn.Sequential(*modules)
    model_backbone.out_channels = out_channels
    anchor_generator = AnchorGenerator(
        sizes=(cfg['model']['anchor_sizes'],),
        aspect_ratios=(cfg['model']['anchor_ratios'],))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'], output_size=7, sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    if 'detector' not in cfg['model'].keys() or (cfg['model']['detector'] == 'faster-rcnn'):
        model = FasterRCNN(
            model_backbone,
            num_classes=cfg['model']['num_classes'],
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            image_mean=[0., 0., 0.],
            image_std=[1., 1., 1.],
        )
    elif cfg['model']['detector'] == 'retinanet':
        model = RetinaNet(
            model_backbone,
            num_classes=cfg['model']['num_classes'],
            anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=224,
            max_size=224,
            image_mean=[0., 0., 0.],
            image_std=[1., 1., 1.],
        )
    else:
        raise Exception(
            f"{cfg['model']['detector']} is not a valid detector. Try faster-rcnn or retinanet")

    model.to(device)

    # training configs
    optimizer = getattr(optim, cfg['training']['optimizer'])
    optimizer = optimizer(model.parameters(), **cfg['training']['optimizer_args'])

    scheduler = getattr(lr_scheduler, cfg['training']['lr_scheduler'])
    scheduler = scheduler(optimizer, **cfg['training']['lr_scheduler_args'])

    # train the model
    train_model(datasets, dataloaders, model, optimizer, scheduler, cfg)


if __name__ == '__main__':
    main()
