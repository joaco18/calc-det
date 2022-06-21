from pathlib import Path
thispath = Path.cwd().resolve()
import sys; sys.path.insert(0, str(thispath.parent))

import logging
import math
import time
import torch
import torchvision

import deep_learning.detection_models.vision_utils.utils as utils
from deep_learning.detection_models.vision_utils.coco_eval import CocoEvaluator
from deep_learning.detection_models.vision_utils.coco_utils import get_coco_api_from_dataset


logging.basicConfig(level=logging.INFO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model, optimizer, data_loader, device, epoch, cfg, scaler=None, tensorboard_writer=None
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    it = 0
    divider = 0
    running_loss = 0
    if cfg['training']['max_iters_per_epoch'] is None:
        max_it = len(data_loader)
    else:
        max_it = cfg['training']['max_iters_per_epoch']

    for images, targets in metric_logger.log_every(
            data_loader, cfg['training']['log_iters'], total=max_it, header=header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # get the epoch loss cumulatively
        running_loss += loss_value * len(images)
        divider += len(images)

        if (it != 0) and ((it % cfg['training']['log_iters']) == 0):
            # compute and log the metrics for the iteration
            iter_loss = running_loss / divider
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("Loss/train_it", iter_loss, it+(max_it*epoch))

        if cfg['training']['max_iters_per_epoch'] is not None:
            if it == cfg['training']['max_iters_per_epoch']:
                break
        it += 1
    loss_value = running_loss / divider
    return metric_logger, loss_value


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluation(model, data_loader, device, coco=None):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    if coco is None:
        coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 1500, header=header):
        images = list(img.to(device) for img in images)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    stats = coco_evaluator.coco_eval['bbox'].stats
    metrics = {
        'AP_IoU_0.50_0.95_all': stats[0],
        'AP_IoU_0.50_all': stats[1],
        'AP_IoU_0.75_all': stats[2],
        'AR_IoU_0.50_0.95_all_mdets_1': stats[6],
        'AR_IoU_0.50_0.95_all_mdets_10': stats[7],
        'AR_IoU_0.50_0.95_all_mdets_100': stats[8],
    }
    torch.set_num_threads(n_threads)
    return coco_evaluator, metrics
