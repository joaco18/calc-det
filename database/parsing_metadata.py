import argparse
import cv2
import logging
import os
import plistlib

import numpy as np
import pandas as pd
import SimpleITK as sitk

from pathlib import Path
from typing import List
from tqdm import tqdm


def load_point(point_string: str):
    return tuple(
        [float(num) for num in point_string.strip('()').split(', ')]
    )


def extract_bbox(points: List[np.ndarray]):
    """
    Generates the coordinates of the bounding box arround
        a lesion decribed by its points.
    Args:
        points (List[np.ndarray]): Points in the lesion contour
    Returns:
        List[np.ndarray]: top-left and right-bottom coords.
    """
    points = np.array(points, dtype='int')
    top_left_point = points.min(axis=0)
    bottom_right_point = points.max(axis=0)
    return [top_left_point, bottom_right_point]


def readxml(filename: Path, image_filanme: str, im_shape: tuple):
    """
    Parse the image metadata xml
    Args:
        filename (Path): Path to the xml file.
        image_filanme (str): Name of the original dcm.
        im_shape (tuple): Dimensions of the image.
    Returns:
        images_row (dict): Containing metadata of the image.
        rois_df (pd.DataFrame): Containing all the metadata associated
            with each lesion.
        mask (np.ndarray): Mask of the lesions, each lesion is
            identified with a numerical label
    """
    # Initialize metadata containers and mask
    rois_df = pd.DataFrame(columns=[
        'patient_id', 'side', 'view', 'Area', 'Center', 'Dev', 'IndexInImage',
        'Max', 'Mean', 'Min', 'Name', 'NumberOfPoints', 'Point_mm', 'Point_px',
        'Total', 'Type', 'lesion_bbox', 'stored'
    ])
    images_row = {}
    mask = np.zeros(im_shape)

    with open(filename, 'rb') as xml:
        # Parse the xml file
        image_dict = plistlib.load(xml, fmt=plistlib.FMT_XML)['Images'][0]

        images_row['n_rois'] = image_dict['NumberOfROIs']
        images_row['patient_id'] = image_filanme.split('_')[0]
        rois = image_dict['ROIs']

        if len(rois) != image_dict['NumberOfROIs']:
            logging.warning(f'Xml {filename} ignored, missing rois')

        # Extract datapoints of the lesion roi
        for roi in rois:
            points = roi['Point_px']
            if roi['NumberOfPoints'] != len(points):
                logging.warning(
                    f'Roi ignored in file {filename}, there are missing points in the ROI'
                )
                continue
            points = [load_point(point) for point in points]
            roi['Point_px'] = points
            roi['Point_mm'] = [load_point(point) for point in roi['Point_mm']]

            # Add other metadata fields
            roi.update({
                'lesion_bbox': extract_bbox(points),
                'patient_id': image_filanme.split('_')[0],
                'side': image_filanme.split('_')[3],
                'view': image_filanme.split('_')[4]
            })

            # Add lesion to the mask
            label = roi['IndexInImage']
            roi['stored'] = True
            if len(points) <= 2:
                for point in points:
                    try:
                        mask[int(point[0]), int(point[1])] = label
                    except Exception as e:
                        logging.warning(
                            f'ROI {roi["IndexInImage"]} of patient '
                            f'{roi["patient_id"]} not stored,'
                            f'coordinates out of boundary. Exception {e}. '
                            f'Point: {point}. Image size {mask.shape}'
                        )
                        roi['stored'] = False
            else:
                cv2.fillPoly(mask, pts=[np.asarray(points, dtype='int')], color=label)
            # Update datafrrame
            rois_df = rois_df.append(pd.Series(roi), ignore_index=True)
    return images_row, rois_df, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ib-path",
        help="path to the base directory of the database *INbreast Release 1.0*")
    parser.add_argument(
        "--masks-dir", const=None,
        help="path to the directory where all the masks are going to be stored")
    parser.add_argument(
        "--pngs-dir", const=None,
        help="path to the directory where all the masks are going to be stored")
    parser.add_argument(
        "--csvs-dir", const=None,
        help="path to the directory where the csvs are going to be stored")
    args = parser.parse_args()

    # Generate paths and directories
    base_path = Path(args.ib_path)
    csvs_path = base_path if args.csvs_dir is None else Path(args.csvs_dir)
    png_folder = base_path/'AllPNGs' if args.pngs_dir is None else Path(args.pngs_dir)
    masks_folder = base_path/'AllMasks' if args.masks_dir is None else Path(args.masks_dir)
    dcm_folder = base_path/'AllDICOMs'
    xml_folder = base_path/'AllXML'

    masks_folder.mkdir(parents=True, exist_ok=True)
    png_folder.mkdir(parents=True, exist_ok=True)

    # Initialize metadata dataframes
    rois_df = pd.DataFrame(columns=[
        'patient_id', 'side', 'view', 'Area', 'Center', 'Dev', 'IndexInImage',
        'Max', 'Mean', 'Min', 'Name', 'NumberOfPoints', 'Point_mm', 'Point_px',
        'Total', 'Type', 'lesion_bbox', 'stored'
    ])
    images_df = pd.DataFrame(columns=['id', 'n_rois', 'side', 'view', 'filename'])

    # Read each dicom and parse the respective xml file
    for filename in tqdm(dcm_folder.iterdir(), total=len(os.listdir(dcm_folder))):
        if filename.suffix != '.dcm':
            continue

        id_ = filename.name.split('_')[0]
        side = filename.name.split('_')[3]
        view = filename.name.split('_')[4]

        # Avoid reprocessing
        mask_filepath = masks_folder / f'{id_}_mask.png'
        if mask_filepath.exists():
            continue

        # Read image
        im = sitk.ReadImage(str(filename))
        im_array = sitk.GetArrayFromImage(im)
        im_array = im_array[0, :, :]

        # Parse xml file
        xml_filepath = xml_folder / f'{id_}.xml'
        if not xml_filepath.exists():
            cv2.imwrite(str(masks_folder/f'{id_}_lesion_mask.png'), np.ones(im_array.shape))
            continue
        images_row, roi_df, mask = readxml(xml_filepath, str(filename.name), im_array.shape)

        # Write mask and pnd version of the image
        cv2.imwrite(str(masks_folder/f'{id_}_lesion_mask.png'), mask)
        png_name = png_folder/f'{id_}.png'
        cv2.imwrite(str(png_name), im_array)
        images_row.update({'side': side, 'view': view, 'filename': png_name})

        # Update dataframes
        rois_df = rois_df.append(roi_df, ignore_index=True)
        images_df = images_df.append(pd.Series(images_row), ignore_index=True)

    # Save metadata in csv
    rois_df.columns = [
        'patient_id', 'side', 'view', 'area', 'center', 'dev', 'index_in_image',
        'max', 'mean', 'min', 'name', 'number_of_points', 'point_mm', 'point_px',
        'total', 'type', 'lesion_bbox', 'stored'
    ]
    images_df.to_csv(csvs_path/'images_metadata.csv')
    rois_df.to_csv(csvs_path/'rois_metadata.csv')


if __name__ == "__main__":
    main()
