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


def load_point(point_string: str, dtype: str = 'float'):
    if dtype == 'float':
        return tuple(
            [float(num) for num in point_string.strip('()').split(', ')]
        )
    elif dtype == 'int':
        return tuple(
            [int(float(num)) for num in point_string.strip('()').split(', ')]
        )
    else:
        raise Exception('dtype not supported to parse points')


def load_muscle_roi_point(point_string: str):
    return tuple(
        [np.maximum(0, int(float(num))) for num in point_string.strip('{}').split(', ')]
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
        bbox_coords (List[np.ndarray]): top-left and right-bottom coords.
            Returned in (x,y) format
        center_coords (tuple): Returned in (x,y) format.
    """
    points = np.array(points, dtype='int')
    bbox_coords = [tuple(points.min(axis=0)), tuple(points.max(axis=0))]
    heigth = bbox_coords[1][1] - bbox_coords[0][1]
    width = bbox_coords[1][0] - bbox_coords[0][0]
    center_coords = \
        (bbox_coords[0][0] + int(width/2), bbox_coords[0][1] + int(heigth/2))
    return bbox_coords, center_coords


def read_pect_musc_xml(filename: Path, im_shape: tuple, side: str):
    """
    Parse the pectoral muscle metadata xml
    Args:
        filename (Path): Path to the xml file.
        im_shape (tuple): Dimensions of the image.
    Returns:
        pectoral_muscle_mask (np.ndarray): Mask of the pectoral_muscle
    """
    label = 255
    pect_muscle_mask = np.zeros(im_shape, dtype='uint8')
    with open(filename, 'rb') as xml:
        # Parse the xml file
        muscle_dict = plistlib.load(xml, fmt=plistlib.FMT_XML)
        if 'Images' in muscle_dict.keys():
            rois = muscle_dict['Images'][0]['ROIs']
            for roi in rois:
                if roi['Name'] != 'Pectoral Muscle':
                    logging.warning(
                        f'Roi ignored in file {filename}, roi name ({roi["Name"]}) '
                        f'not pectoral muscle')
                    continue
                points = roi['Point_px']
                if roi['NumberOfPoints'] != len(points):
                    logging.warning(
                        f'Roi ignored in file {filename}, there are missing points in the ROI')
                    continue
                corner = (im_shape[1], 0) if side == 'R' else (0, 0)
                points = [corner] + [load_point(point, 'int') for point in points]
                if len(points) <= 2:
                    for point in points:
                        try:
                            pect_muscle_mask[int(point[1]), int(point[0])] = label
                        except Exception as e:
                            logging.warning(
                                f'ROI {roi["IndexInImage"]} of image '
                                f'{roi["img_id"]} not stored,'
                                f'coordinates out of boundary. Exception {e}. '
                                f'Point: {point}. Image size {pect_muscle_mask.shape}'
                            )
                else:
                    cv2.fillPoly(
                        pect_muscle_mask, pts=[np.asarray(points, dtype='int')], color=label)
        elif 'ROIPoints' in muscle_dict.keys():
            corner = (im_shape[1], 0) if side == 'R' else (0, 0)
            roi_points = \
                [corner] + [load_muscle_roi_point(point) for point in muscle_dict['ROIPoints']]
            if len(roi_points) <= 2:
                for point in roi_points:
                    try:
                        pect_muscle_mask[int(point[1]), int(point[0])] = label
                    except Exception as e:
                        logging.warning(
                            f'ROI {roi["IndexInImage"]} of image '
                            f'{roi["img_id"]} not stored,'
                            f'coordinates out of boundary. Exception {e}. '
                            f'Point: {point}. Image size {pect_muscle_mask.shape}'
                        )
            else:
                cv2.fillPoly(
                    pect_muscle_mask, pts=[np.asarray(roi_points, dtype='int')], color=label)
    return pect_muscle_mask


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
        'case_id', 'img_id', 'side', 'view', 'Area', 'Center', 'Center_crop', 'Dev',
        'IndexInImage', 'Max', 'Mean', 'Min', 'NumberOfPoints', 'Point_mm', 'Point_px',
        'Point_px_crop', 'Total', 'Type', 'lesion_bbox', 'lesion_bbox_crop', 'stored',
        'acr', 'birads', 'finding_notes', 'lesion_annot', 'pectoral_muscle',
        'artifact', 'lesion_type', 'radius', 'partition'
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
            logging.warning(f'Xml {filename} has missing rois')

        # Extract datapoints of the lesion roi and order them
        # in descending order of size (to avoid maskin of ovelapping rois)
        for roi in rois:
            points = roi['Point_px']
            if roi['NumberOfPoints'] != len(points):
                logging.warning(
                    f'Roi ignored in file {filename}, there are missing points in the ROI'
                )
                continue
            # The points in the XML file are stored in (x,y) format, opencv needs this format
            points = [load_point(point, 'int') for point in points]
            roi['Point_px'] = points
            roi['Point_mm'] = [load_point(point) for point in roi['Point_mm']]
            roi['lesion_type'] = roi['Name']
            del(roi['Name'])
            roi['lesion_bbox'], roi['Center'] = extract_bbox(points)
            roi['IndexInImage'] = roi['IndexInImage'] + 1

            # Add metadata coming from excel
            roi.update(metadata_img)

            roi['stored'] = True
            roi['Area'] = cv2.contourArea(np.array(points))
            rois_df = rois_df.append(pd.Series(roi), ignore_index=True)
        rois_df.sort_values('Area', ascending=False, inplace=True, ignore_index=False)

        for indx, roi in rois_df.iterrows():
            # Add lesion to the mask
            label = roi['IndexInImage']
            points = roi['Point_px']
            if len(points) <= 2:
                for point in points:
                    try:
                        mask[int(point[1]), int(point[0])] = label
                    except Exception as e:
                        logging.warning(
                            f'ROI {roi["IndexInImage"]} of image '
                            f'{roi["img_id"]} not stored,'
                            f'coordinates out of boundary. Exception {e}. '
                            f'Point: {point}. Image size {mask.shape}'
                        )
                        rois_df.loc[indx, 'stored'] = False
            else:
                cv2.fillPoly(mask, pts=[np.asarray(points, dtype='int')], color=label)
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
    la = row.lesion_annotation_status
    lesion_annot = 'no-normal' \
        if isinstance(la, str) and (la.lower() == 'no annotation (normal)') else 'yes'
    pm = row.pectoral_muscle_annotation
    pectoral_muscle = \
        isinstance(pm, str) and (pm.lower() != 'without muscle') and (row.view == 'MLO')
    art = row.other_annotations
    artifact = isinstance(art, str) and art.lower() == 'artifact not annotated'
    return {
        'img_id': img_id,
        'side':  row.laterality,
        'view': row['view'],
        'acr': row.acr,
        'birads': str(row['bi-rads']),
        'mass': row.mass_ == 'X',
        'micros': row.micros == 'X',
        'distortion': row.distortion == 'X',
        'asymmetry': row.asymmetry == 'X',
        'finding_notes': row['findings_notes_(in_portuguese)'],
        'lesion_annot': lesion_annot,
        'pectoral_muscle': pectoral_muscle,
        'artifact': artifact,
    }


def get_breast_bbox(image: np.ndarray):
    """
    Makes a threshold of the image identifying the regions different from
    the background (0). Takes the largest (area) region (the one corresponding
    to the breast), defines the contour of this region and creates a roi
    that fits it.

    Args:
        image (np.ndarray): Breast image to be croped.
    Return:
        breast_bbox (List[tuple]): Coordinates of the bounding box.
            [(x, y), (x+w, y+h)] (topleft, rightbottom)
        mask (np.ndarray): binary mask image of the breast.
    """

    # Threshold image with th=0 and get connected comp.
    img = image.copy()
    img[img != 0] = 255
    img = img.astype('uint8')
    nb_components, output, stats, _ = \
        cv2.connectedComponentsWithStats(img, connectivity=4)

    # Get the areas of each connected component
    sizes = stats[:, -1]
    # Keep the largest connected component excluding bkgd
    max_label = np.argmax(sizes[1:]) + 1
    x, y, w, h = stats[max_label, 0:4]
    breast_bbox = [(x, y), (x+w, y+h)]

    # Generate a binary mask for the breast
    mask = np.zeros(img.shape)
    mask[output == max_label] = 1
    return breast_bbox, mask


def update_rois_coords(roi_df: pd.DataFrame, breast_bbox: List[tuple]):
    """
    Uses the origin of the bbox of the breast to update the coordinates
    of the different points in the dataframe.
    Args:
        roi_df (pd.DataFrame): df including all roi's metadata.
        breast_bbox (List[tuple]): Coordinates of the bounding box.
            [(x, y), (x+w, y+h)] (topleft, rightbottom).
    Returns:
        out_df (pd.DataFrame): df with the coordinates in the cropped
            image.
    """
    out_df = roi_df.copy()
    [x_ori, y_ori] = np.array(breast_bbox[0])
    for new_column in ['Point_px_crop', 'lesion_bbox_crop', 'Center_crop']:
        out_df[new_column] = np.nan
        out_df[new_column] = out_df[new_column].astype('object')
    for row in roi_df.iterrows():
        out_df.at[row[0], 'Point_px_crop'] = \
            [(point[0] - x_ori, point[1] - y_ori) for point in row[1]['Point_px']]
        out_df.at[row[0], 'lesion_bbox_crop'] = \
            [(point[0] - x_ori, point[1] - y_ori) for point in row[1]['lesion_bbox']]
        out_df.at[row[0], 'Center_crop'] = \
            (row[1]['Center'][0] - x_ori, row[1]['Center'][1] - y_ori)
        _, out_df.at[row[0], 'radius'] = cv2.minEnclosingCircle(
            np.asarray(out_df.at[row[0], 'Point_px'])
        )
    return out_df


def format_roi_df(rois_df: pd.DataFrame):
    """
    Fixes lesion type values, and turn columns names into snake case
    Args:
        rois_df (pd.DataFrame)
    Returns:
        rois_df (pd.DataFrame)
    """
    rois_df.columns = [to_snake_case(name) for name in rois_df.columns]
    replacements = {
        'Calcification': 'calcification',
        'Cluster': 'cluster',
        'Mass': 'mass',
        'Point 3': 'calcification',
        'Espiculated Region': 'spiculated_region',
        'Spiculated Region': 'spiculated_region',
        'Distortion': 'distortion',
        'Asymmetry': 'asymmetry',
        'Unnamed': 'calcification',
        'Point 1': 'unkown',
        'Calcifications': 'cluster',
        'Assymetry': 'asymmetry',
        'Spiculated region': 'spiculated_region',
    }
    rois_df.fillna('unknown', inplace=True)
    rois_df.replace({"lesion_type": replacements}, inplace=True)
    return (rois_df)


def add_image_and_case_label(img_df: pd.DataFrame):
    """
    Based on the specific lesions present in the image add a normal abnormal
    label column for the study and for the image.
    Args:
        img_df (pd.DataFrame)
    Returns:
        (pd.DataFrame)
    """
    img_df['case_label'] = 'normal'
    img_df['img_label'] = 'normal'

    pathologic = (
        (img_df.mass) | (img_df.micros) |
        (img_df.distortion) | (img_df.asymmetry)
    )
    pathologic_studies = img_df.loc[pathologic, 'case_id'].unique()
    img_df.loc[img_df.case_id.isin(pathologic_studies), 'case_label'] = 'abnormal'
    img_df.loc[pathologic, 'img_label'] = 'abnormal'
    return img_df


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
        '--v', dest='verbose', action='store_true',
        help='Wheter to print the warnings or not')
    parser.add_argument(
        '--cb', dest='crop_breast', action='store_true',
        help='Just save the breast region in the pngs.')
    parser.add_argument(
        '--pect-musc-mask', dest='pectoral_muscle_mask', action='store_true',
        help='Retrieve the pectoral muscle mask')
    args = parser.parse_args()

    if not bool(args.verbose):
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    # Generate paths and directories
    base_path = Path(args.ib_path)
    csvs_path = base_path if args.csvs_dir is None else Path(args.csvs_dir)
    png_folder = base_path/'AllPNGs'/'full_imgs' if args.pngs_dir is None else Path(args.pngs_dir)
    masks_folder = \
        base_path/'AllMasks'/'full_imgs' if args.masks_dir is None else Path(args.masks_dir)
    muscle_masks_folder = base_path / 'AllMasks'/'muscle_masks'
    dcm_folder = base_path / 'AllDICOMs'
    xml_folder = base_path / 'AllXML'
    muscle_xml_folder = base_path / 'PectoralMuscle' / 'Pectoral Muscle XML'

    masks_folder.mkdir(parents=True, exist_ok=True)
    png_folder.mkdir(parents=True, exist_ok=True)
    muscle_masks_folder.mkdir(parents=True, exist_ok=True)

    # Load and adjust the provided metadata dataframe
    df_excel = pd.read_excel(base_path / 'INbreast.xls')
    df_excel = df_excel.iloc[:-2, :]
    df_excel.columns = [column.replace(' ', '_').lower() for column in df_excel.columns]
    df_excel.loc[:, 'file_name'] = df_excel.file_name.astype('int').astype('str')

    # Initialize metadata dataframes
    rois_df = pd.DataFrame(columns=[
        'case_id', 'img_id', 'side', 'view', 'Area', 'Center', 'Center_crop', 'Dev',
        'IndexInImage', 'Max', 'Mean', 'Min', 'NumberOfPoints', 'Point_mm', 'Point_px',
        'Point_px_crop', 'Total', 'Type', 'lesion_bbox', 'lesion_bbox_crop', 'stored',
        'acr', 'birads', 'mass', 'micros', 'distortion', 'asymmetry', 'finding_notes',
        'lesion_annot', 'pectoral_muscle', 'artifact', 'lesion_type', 'radius', 'partition'
    ])

    images_df = pd.DataFrame(columns=[
        'img_id', 'n_rois', 'side', 'view', 'filename', 'acr', 'artifact', 'birads',
        'case_id', 'finding_notes', 'lesion_annot', 'pectoral_muscle', 'mass',
        'micros', 'distortion', 'asymmetry', 'breast_bbox', 'partition', 'img_size',
        'muscle_mask'
    ])

    # Load the predefined training set
    train_set_images = pd.read_csv('data/standard_partitions.csv')
    train_set_images = train_set_images.loc[
        train_set_images.partition == 'train', 'image_id'].astype(str).tolist()

    # Read each dicom and parse the respective xml file
    for filename in tqdm(dcm_folder.iterdir(), total=len(os.listdir(dcm_folder))):
        if filename.suffix != '.dcm':
            continue
        img_id = filename.name.split('_')[0]
        metadata_img = parse_metadata_from_excel(img_id, df_excel)
        metadata_img['case_id'] = filename.name.split('_')[1]

        # Generate partition label:
        metadata_img['partition'] = 'train' if img_id in train_set_images else 'test'

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
        if xml_filepath.exists():
            images_row, roi_df, mask = readxml(xml_filepath, metadata_img, im_array.shape)
            for row in roi_df.iterrows():
                _, roi_df.at[row[0], 'radius'] = cv2.minEnclosingCircle(
                    np.asarray(roi_df.at[row[0], 'Point_px']))
        else:
            images_row = {'n_rois': 0, 'breast_bbox': None}
            images_row.update(metadata_img)

        pectoral_muscle_available = False
        if args.pectoral_muscle_mask:
            pect_musc_xml_filepath = muscle_xml_folder / f'{img_id}_muscle.xml'
            if pect_musc_xml_filepath.exists():
                pectoral_muscle_available = True
                pect_musc_mask = read_pect_musc_xml(
                    pect_musc_xml_filepath, im_array.shape, metadata_img['side'])
        images_row['muscle_mask'] = pectoral_muscle_available

        if args.crop_breast:
            # If indicated extract the breast bbox and update rois coords
            bbox, _ = get_breast_bbox(im_array.copy())
            im_array = im_array[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
            mask = mask[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
            if pectoral_muscle_available:
                pect_musc_mask = pect_musc_mask[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
            if xml_filepath.exists():
                roi_df = update_rois_coords(roi_df.copy(), bbox.copy())
            images_row['breast_bbox'] = bbox

        # Update dataframes and write mask and png version of the image
        png_name = png_folder/f'{img_id}.png'
        images_row['filename'] = png_name
        images_row['img_size'] = im_array.shape

        images_df = images_df.append(pd.Series(images_row), ignore_index=True)
        # im_array = (im_array - im_array.min()) / (im_array.max() - im_array.min()) * 255
        # im_array = im_array.astype('uint8')
        cv2.imwrite(str(png_name), im_array)

        if pectoral_muscle_available:
            cv2.imwrite(
                str(muscle_masks_folder/f'{img_id}_pectoral_muscle_mask.png'), pect_musc_mask)

        if xml_filepath.exists():
            rois_df = rois_df.append(roi_df, ignore_index=True)
            cv2.imwrite(str(masks_folder/f'{img_id}_lesion_mask.png'), mask)

    # Save metadata in csv
    rois_df = format_roi_df(rois_df)
    rois_df.sort_values(by='img_id', inplace=True)
    images_df = add_image_and_case_label(images_df)
    images_df.sort_values(by='img_id', inplace=True)
    images_df.to_csv(csvs_path/'images_metadata.csv')
    rois_df.to_csv(csvs_path/'rois_metadata.csv')


if __name__ == "__main__":
    main()
