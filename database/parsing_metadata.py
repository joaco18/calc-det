import argparse
import cv2
import logging
import os
import plistlib
import re

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


def to_snake_case(string: str):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', string).lower()


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
    return [tuple(points.min(axis=0)), tuple(points.max(axis=0))]


def readxml(filename: Path, metadata_img: dict, im_shape: tuple):
    """
    Parse the image metadata xml
    Args:
        filename (Path): Path to the xml file.
        metadata_img (dict): Dictionary containing the metadata included
            in the excel file.
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
        'img_id', 'side', 'view', 'Area', 'Center', 'Dev', 'IndexInImage',
        'Max', 'Mean', 'Min', 'NumberOfPoints', 'Point_mm', 'Point_px',
        'Total', 'Type', 'lesion_bbox', 'stored'
    ])
    images_row = {}
    mask = np.zeros(im_shape)

    with open(filename, 'rb') as xml:
        # Parse the xml file
        image_dict = plistlib.load(xml, fmt=plistlib.FMT_XML)['Images'][0]

        images_row['n_rois'] = image_dict['NumberOfROIs']
        images_row.update(metadata_img)
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
            roi['Center'] = load_point(roi['Center'])
            roi['lesion_type'] = roi['Name']
            del(roi['Name'])
            roi['lesion_bbox'] = extract_bbox(points)
            roi['IndexInImage'] = roi['IndexInImage'] + 1

            # Add metadata coming from excel
            roi.update(metadata_img)

            # Add lesion to the mask
            label = roi['IndexInImage'] + 1
            roi['stored'] = True
            if len(points) <= 2:
                for point in points:
                    try:
                        mask[int(point[0]), int(point[1])] = label
                    except Exception as e:
                        logging.warning(
                            f'ROI {roi["IndexInImage"]} of image '
                            f'{roi["img_id"]} not stored,'
                            f'coordinates out of boundary. Exception {e}. '
                            f'Point: {point}. Image size {mask.shape}'
                        )
                        roi['stored'] = False
            else:
                cv2.fillPoly(mask, pts=[np.asarray(points, dtype='int')], color=label)
            # Update datafrrame
            rois_df = rois_df.append(pd.Series(roi), ignore_index=True)
    return images_row, rois_df, mask


def parse_metadata_from_excel(img_id: str, df_excel: pd.DataFrame):
    """
    Gets metadata from the excel file
    Args:
        img_id (str): id of the image analysed
        df_excel (pd.DataFrame): Excel metadata dataframe
    Returns:
        (dict): Containing desired metadata.
    """
    row = df_excel.loc[df_excel.file_name == img_id, :].squeeze()
    # Following a binary code in this order: (mass, micro, dist, asym)
    type_excel = (
        row.mass_ == 'X',
        row.micros == 'X',
        row.distortion == 'X',
        row.asymmetry == 'X'
    )
    la = row.lesion_annotation_status
    lesion_annot = 'no-normal' \
        if isinstance(la, str) and (la.lower() == 'no annotation (normal)') else 'yes'
    pm = row.pectoral_muscle_annotation
    pectoral_muscle = \
        isinstance(pm, str) and (pm.lower() != 'without muscle') and (row.view == 'MLO')
    art = row.other_annotations
    artifact = isinstance(art, str) and art.lower() == 'artifact not annotated'
    acr = int(row.acr) if isinstance(row.acr, float) else None
    return {
        'img_id': img_id,
        'side':  row.laterality,
        'view': row['view'],
        'acr': acr,
        'birads': str(row['bi-rads']),
        'type_excel': type_excel,
        'finding_notes': row['findings_notes_(in_portuguese)'],
        'lesion_annot': lesion_annot,
        'pectoral_muscle': pectoral_muscle,
        'artifact': artifact,
    }


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
    parser.add_argument(
        '--rp', dest='reprocess', action='store_true',
        help='Whether to reprocess the case if the mask already exists or not')
    parser.add_argument(
        '--v', dest='verbose', action='store_true', help='Wheter to print the warnings or not')
    args = parser.parse_args()

    if not bool(args.verbose):
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    # Generate paths and directories
    base_path = Path(args.ib_path)
    csvs_path = base_path if args.csvs_dir is None else Path(args.csvs_dir)
    png_folder = base_path / 'AllPNGs' if args.pngs_dir is None else Path(args.pngs_dir)
    masks_folder = base_path / 'AllMasks' if args.masks_dir is None else Path(args.masks_dir)
    dcm_folder = base_path / 'AllDICOMs'
    xml_folder = base_path / 'AllXML'

    masks_folder.mkdir(parents=True, exist_ok=True)
    png_folder.mkdir(parents=True, exist_ok=True)

    # Load and adjust the provided metadata dataframe
    df_excel = pd.read_excel(base_path / 'INbreast.xls')
    df_excel = df_excel.iloc[:-2, :]
    df_excel.columns = [column.replace(' ', '_').lower() for column in df_excel.columns]
    df_excel.loc[:, 'file_name'] = df_excel.file_name.astype('int').astype('str')

    # Initialize metadata dataframes
    rois_df = pd.DataFrame(columns=[
        'case_id', 'img_id', 'side', 'view', 'Area', 'Center', 'Dev', 'IndexInImage',
        'Max', 'Mean', 'Min', 'NumberOfPoints', 'Point_mm', 'Point_px',
        'Total', 'Type', 'lesion_bbox', 'stored', 'acr', 'birads', 'type_excel',
        'finding_notes', 'lesion_annot', 'pectoral_muscle', 'artifact', 'lesion_type'
    ])

    images_df = pd.DataFrame(columns=[
        'img_id', 'n_rois', 'side', 'view', 'filename', 'acr', 'artifact', 'birads',
        'case_id', 'finding_notes', 'lesion_annot', 'pectoral_muscle', 'type_excel'
    ])

    # Read each dicom and parse the respective xml file
    k = 0
    for filename in tqdm(dcm_folder.iterdir(), total=len(os.listdir(dcm_folder))):
        if filename.suffix != '.dcm':
            continue
        img_id = filename.name.split('_')[0]
        metadata_img = parse_metadata_from_excel(img_id, df_excel)
        metadata_img['case_id'] = filename.name.split('_')[1]

        # Avoid reprocessing
        mask_filepath = masks_folder / f'{img_id}_mask.png'
        if not args.reprocess and mask_filepath.exists():
            continue

        # Read image
        im = sitk.ReadImage(str(filename))
        im_array = sitk.GetArrayFromImage(im)
        im_array = im_array[0, :, :]

        # Parse xml file
        xml_filepath = xml_folder / f'{img_id}.xml'
        if not xml_filepath.exists():
            cv2.imwrite(str(masks_folder/f'{img_id}_lesion_mask.png'), np.zeros(im_array.shape))
            continue
        images_row, roi_df, mask = readxml(xml_filepath, metadata_img, im_array.shape)

        # Write mask and pnd version of the image
        cv2.imwrite(str(masks_folder/f'{img_id}_lesion_mask.png'), mask)
        png_name = png_folder/f'{img_id}.png'
        cv2.imwrite(str(png_name), im_array)
        images_row.update(metadata_img)
        images_row['filename'] = png_name

        # Update dataframes
        rois_df = rois_df.append(roi_df, ignore_index=True)
        images_df = images_df.append(pd.Series(images_row), ignore_index=True)

    # Save metadata in csv
    rois_df.columns = [to_snake_case(name) for name in rois_df.columns]
    images_df.to_csv(csvs_path/'images_metadata.csv')
    rois_df.to_csv(csvs_path/'rois_metadata.csv')


if __name__ == "__main__":
    main()
