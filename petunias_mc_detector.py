import argparse
import logging
import time
import torch

import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path

from machine_learning.ml_detector import DetectorML
from deep_learning.classification_models.models.class_based_detector import \
    ClassificationBasedDetector
from deep_learning.detection_models.models.detect_based_detector import DetectionBasedDetector
from database.parsing_metadata import get_breast_bbox
import deep_learning.dl_utils as dl_utils
import general_utils.utils as utils


root_path = Path.cwd()
pd.options.mode.chained_assignment = None
logging.basicConfig(level=logging.ERROR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcm-filepath", help="Path to the dicom file to process")
    parser.add_argument(
        "--detector-type", help="One of: ['aia_ml', 'classification_dl', 'detection_dl']")
    parser.add_argument(
        "--batch-size", default=0, help="Size of batch to use in dl")
    parser.add_argument(
        "--ouput-path", help="Directory where the detections masks is going to be stored")
    parser.add_argument(
        "--store-csv", action='store_true',
        help="Whether to store a csv file with the detections in the same path as output image")
    parser.add_argument(
        '--v', dest='verbose', action='store_true', help='Wheter to print process info or not')
    args = parser.parse_args()

    if not bool(args.verbose):
        logging.getLogger().setLevel(logging.ERROR)
    else:
        logging.getLogger().setLevel(logging.INFO)

    assert args.detector_type in ['aia_ml', 'classification_dl', 'detection_dl'], \
        'Detector type not supported try one of : ' \
        '\'aia_ml\', \'classification_dl\', \'detection_dl\''

    start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info(f'Instantiating the model of type : {args.detector_type}')
    # Get the detector
    if args.detector_type == 'aia_ml':
        model_chkpt_path = str(root_path / 'machine_learning/checkpoints/cascade_models.pkl')
        detector = DetectorML(model_chkpt_path)
        threshold = 0.13444709378395178

    elif args.detector_type == 'classification_dl':
        assert device.type == "cuda", 'Deep learning models are only supported for GPU runtime'
        model_path = str(
              root_path / 'deep_learning/classification_models/checkpoints/16_net2_07.pt')
        model_ckpt = torch.load(model_path)
        model = dl_utils.get_model_from_checkpoint(model_ckpt)
        model.eval()

        bs = 2048 if args.batch_size == 0 else args.batch_size
        detector = ClassificationBasedDetector(
            model, threshold=0.21540279686450958, pred_kind='score', norm_kind='avg',
            post_proc=True, k_size=9, patch_size=16, stride=8, min_breast_fraction_patch=0.7,
            batch_size=bs, device=device, nms=True, iou_threshold=1
        )
        threshold = 0.21540279686450958

    else:
        assert device.type == "cuda", 'Deep learning models are only supported for GPU runtime'
        # detector instatiation
        ckpt_path = str(
            root_path / 'deep_learning/detection_models/checkpoints/d_resnet50_00.pt')
        backbone_path = str(
            root_path / 'deep_learning/classification_models/checkpoints/224_resnet50_05.pt')
        mdl_ckpt = torch.load(ckpt_path)
        mdl_ckpt['configuration']['model']['checkpoint_path'] = backbone_path
        bs = 24 if args.batch_size == 0 else args.batch_size
        detector = DetectionBasedDetector(
            model_chkp=mdl_ckpt, patch_size=224, stride=200, min_breast_fraction_patch=0.7,
            batch_size=bs, bbox_size=14, device=device, iou_threshold=0.5,
            score_threshold=0.8870506882667542)
        threshold = 0.8870506882667542

    logging.info('Reading the image...')
    # get the image
    im = sitk.ReadImage(str(args.dcm_filepath))
    im_array = sitk.GetArrayFromImage(im)
    im_array = im_array[0, :, :]

    # crop the breast region and turn them to left ones
    bbox_orig, breast_mask = get_breast_bbox(im_array.copy())
    right = utils.is_right(breast_mask)
    bbox = bbox_orig
    if right:
        im_array = np.fliplr(im_array)
        bbox, _ = get_breast_bbox(im_array.copy())
    im_array = im_array[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]

    # get the detections
    logging.info('Detecting...')
    detections = detector.detect(im_array)

    # postprocess the detections
    if args.detector_type == 'aia_ml':
        detections_centers_and_radius = np.stack(
            detections['candidate_coordinates'].values).astype(int)
        detections_scores = detections['confidence'].values.reshape(-1, 1)
        # retain the centers and scores:
        detections = np.concatenate(
            [detections_centers_and_radius, detections_scores], axis=1)
    else:
        detections_centers = detections[:, 0:2].astype(int)
        detections_scores = detections[:, -1].reshape(-1, 1)
        detections_radius = np.ones(detections_scores.shape) * 7
        detections = np.concatenate(
            [detections_centers, detections_radius, detections_scores], axis=1)

    detections_df = pd.DataFrame(detections, columns=['x', 'y', 'radius', 'score'])
    detections_df = detections_df.loc[detections_df.score >= threshold, :]

    # store detections as dcm mask
    logging.info('Storing...')
    dcm_path = Path(args.dcm_filepath)
    dcm_name = dcm_path.stem
    output_filepath = Path(args.ouput_path) / f'{dcm_name}_{args.detector_type}_detections.dcm'

    detections_df = pd.DataFrame(detections, columns=['x', 'y', 'radius', 'score'])
    utils.store_as_dcm(im_array, detections_df, dcm_path, output_filepath, tuple(bbox_orig))

    if args.store_csv:
        csv_filepath = Path(args.ouput_path) / f'{dcm_name}_{args.detector_type}_detections.csv'
        detections_df.to_csv(csv_filepath)
    total_time = time.time() - start
    logging.info(f'Detection finished. It took {total_time:.4}s')


if __name__ == '__main__':
    main()
