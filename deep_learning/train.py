from pathlib import Path
thispath = Path.cwd().resolve()
import sys; sys.path.insert(0, str(thispath.parent))

from deep_learning.dataset.dataset import INBreast_Dataset_pytorch
from deep_learning.models.base_classifier import CNNClasssifier
import deep_learning.dl_utils as dl_utils

import logging
import torch
import time
import random
import yaml

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def identity_function(arg):
    return arg


def train_model(datasets, dataloaders, data_transforms, model, criterion, optimizer, scheduler, cfg):

    # guarantee reproducibility
    since = time.time()
    random.seed(0)
    torch.manual_seed(1442)
    np.random.seed(0)

    # holders for best model
    best_metric = 0.0
    best_metric_name = cfg['training']['best_metric']

    exp_path = Path.cwd().parent.parent/f'data/deepl_runs/{cfg["experiment_name"]}'
    exp_path.mkdir(exist_ok=True, parents=True)
    best_model_path = exp_path / f'{cfg["experiment_name"]}.pt'
    chkpt_path = exp_path / f'{cfg["experiment_name"]}_chkpt.pt'

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

    for epoch in range(init_epoch, cfg['num_epochs']):
        logging.info(f'Epoch {epoch+1}/{cfg["num_epochs"]}')
        logging.info(('-' * 10))

        # resample used negatives to use the large diversity of them that we have
        datasets['train'].update_sample_used(epoch)
        dataloaders['train'] = DataLoader(
            datasets['train'], batch_size=cfg['dataloaders']['train_batch_size'],
            shuffle=True, num_workers=4, drop_last=False)

        for phase in ['train', 'val']:
            # Set model to the corresponding mode and update lr if necessary
            if phase == 'train':
                if epoch != 0:
                    scheduler.step()
                writer.add_scalar(f"LearningRate/{phase}", scheduler.get_last_lr()[0], epoch)
                model.train()
            else:
                model.eval()

            # define holders for losses, preds and labels
            running_loss = 0.0
            epoch_preds, epoch_labels = [], []

            # Iterate over data.
            n_data = len(dataloaders[phase])
            for it, sample in tqdm(enumerate(dataloaders[phase]), total=n_data):

                # Apply transformations and send to device
                sample['img'] = data_transforms[phase](sample['img'])
                inputs = sample['img'].to(device)
                labels = sample['label'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass (track history if only in train)
                with torch.set_grad_enabled(phase == 'train'):
                    # predict
                    outputs = model(inputs)

                    # store values
                    epoch_preds.append(np.asarray(
                        torch.sigmoid(outputs.detach()).flatten().cpu()))
                    epoch_labels.append(np.asarray(labels.detach().cpu()))

                    # finish the comp. graph
                    loss = criterion(outputs.flatten(), labels.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # once in a while store the images batch to check
                if it in [25, 50, 100]:
                    imgs = T.functional.rgb_to_grayscale(sample['img']).cpu()
                    writer.add_images(f'Images/{phase}', imgs, epoch)
                    del imgs

                # get the epoch loss cumulatively
                running_loss += loss.item() * inputs.size(0)

            # compute and log the metrics for the epoch
            epoch_preds = np.concatenate(epoch_preds)
            epoch_labels = np.concatenate(epoch_labels)
            epoch_loss = running_loss / len(epoch_preds)
            metrics = dl_utils.get_metrics(epoch_labels, epoch_preds)
            dl_utils.tensorboard_logs(writer, epoch_loss, epoch, metrics, phase)

            # print status
            epoch_f1 = metrics['f1_score']
            message = f'{phase} Loss: {epoch_loss:.4f} Acc: {metrics["accuracy"]:.4f}' \
                f' F1: {epoch_f1:.4f} AUROC: {metrics["auroc"]:.4f}'
            logging.info(message)

            # save last and best checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, chkpt_path)

            if phase == 'val' and metrics[best_metric_name] > best_metric:
                best_metric = epoch_f1
                best_threshold = metrics['threshold']
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'metrics': metrics,
                    'configuration': cfg
                    }, best_model_path)
        logging.info()

    time_elapsed = time.time() - since
    message = f'Training complete in {(time_elapsed // 60):.0f}m ' \
        f'{(time_elapsed % 60):.0f}s'
    logging.info(message)
    logging.info(f'Best val {best_metric_name}: {best_metric:4f}, threshold {best_threshold:.4f}')

    # close the tensorboard session
    writer.flush()
    writer.close()

    # load best model weights before returning
    best_model = torch.load(best_model_path)
    model.load_state_dict(best_model['model_state_dict'])
    return model


def main():
    # read the configuration file
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    # use the configuration for the dataset
    dataset_arguments = cfg['dataset']
    dataset_arguments['patch_images_path'] = Path(dataset_arguments['patch_images_path'])
    datasets = {
        'train': INBreast_Dataset_pytorch(
            partitions=['train'], neg_to_pos_ratio=dataset_arguments['train_neg_to_pos_ratio'],
            balancing_seed=0, **dataset_arguments),
        'val': INBreast_Dataset_pytorch(
            partitions=['validation'], neg_to_pos_ratio=None, **dataset_arguments)
    }

    # use the configuration for the dataloaders
    dataloaders = {
        'val': DataLoader(
            datasets['val'], batch_size=cfg['dataloaders']['train_batch_size'],
            num_workers=4, drop_last=False),
        'train': DataLoader(
            datasets['train'], batch_size=cfg['dataloaders']['val_batch_size'],
            shuffle=True, num_workers=4, drop_last=False)
    }

    # use the configuration for the transformations
    transforms = nn.Sequential(
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0, hue=0),
        T.RandomAffine(
            degrees=(0, 20), translate=None, scale=None, shear=(1, 10, 1, 10),
            interpolation=T.InterpolationMode.BILINEAR, fill=0
        ),
        T.RandomPerspective(distortion_scale=0.2),
        T.RandomRotation(degrees=(0, 20)),
        T.RandomRotation(degrees=(90, 110)),
        T.RandomResizedCrop(size=(224, 224), scale=(0.9, 1), ratio=(1, 1)),
        T.RandomAutocontrast(),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip()
    )
    transforms = T.RandomApply(transforms=transforms, p=cfg['data_aug']['prob'])
    data_transforms = {
        'train': identity_function if (cfg['data_aug']['prob'] is None) else transforms,
        'val': identity_function
    }

    # model configs
    model = CNNClasssifier(
        activation=getattr(nn, cfg['model']['activation'])(),
        dropout=cfg['model']['dropout'],
        fc_dims=cfg['model']['fc_dims'],
        freeze_weights=cfg['model']['freeze_weights'],
        backbone=cfg['model']['backbone'],
        pretrained=cfg['model']['pretrained'],
    )
    model = model.model.to(device)

    # training configs
    criterion = getattr(nn, cfg['training']['criterion'])()

    optimizer = getattr(optim, cfg['training']['optimizer'])
    optimizer = optimizer(model.parameters(), **cfg['training']['optimizer_args'])

    scheduler = getattr(lr_scheduler, cfg['training']['lr_scheduler'])
    optimizer = scheduler(optimizer, **cfg['training']['lr_scheduler_args'])

    # train the model
    train_model(datasets, dataloaders, data_transforms, model, criterion, optimizer, scheduler, cfg)


if __name__ == '__main__':
    main()
