import argparse
import cv2
import logging
import os
import pydicom

import numpy as np
import pandas as pd
import SimpleITK as sitk
import xml.etree.ElementTree as ET

from pathlib import Path
from skimage.morphology import disk


def parse_string_tuple(string: str):
    """
    Cast a string version of a tuple to a list of floats
    Args:
        string (str): string of floats e.g (0.0, 0.0, 0.0)
    Returns:
        list: list of floats e.g. [0.0, 0.0, 0.0]
    """
    string = string.rstrip(')')
    string = string.lstrip('(')
    return (np.asarray(string.split(', ')).astype('float')).tolist()


def sanity_check_wz(points: list, width: int, height: int):
    """
    Checks that the points recovered from the xml file don't excede the
        image boundaries
    Args:
        points (list): List of points, must be pixels coordinates
        width (int): Width of the image
        height (int): Height of the image
    Returns:
        minx (int): Minimum row position in of the bbox.
        maxx (int): Maximum row position in of the bbox.
        miny (int): Minimum column position in of the bbox.
        maxy (int): Maximum column position in of the bbox.
        status (bool): Succesfulness
    """
    [minx, miny] = np.asarray(points).min(axis=0).astype(int)
    [maxx, maxy] = np.asarray(points).max(axis=0).astype(int)
    if maxx > width:
        maxx = width
    if maxy > height:
        maxy = height
    if (minx < 0) | (miny < 0):
        status = False
    status = True
    return minx, maxx, miny, maxy, status


def add_lesion_mask(mask: np.ndarray, points: list):
    """
    Add the points of the newly identified lesion  to the preexisting mask
    Args:
        mask (np.ndarray): Lesion contour mask.
        points (list): Points of the boudary of the lesion.
    Returns:
        mask (np.ndarray): Updated lesion contour mask.
    """

    for point in points:
        [xx, yy] = point
        if xx > mask.shape[1]:
            xx = mask.shape[1]
        if yy > mask.shape[0]:
            yy = mask.shape[0]
        mask[xx, yy] = 255
    return mask


def parse_roi_dictionary(roi_dict: ET.Element, height: int, width: int):
    """
    Parses the content of the dictionary element of a roi.
    Args:
        roi_dict (ET.Element): Dictionary element from the image xml.
        height (int): Number of rows in the image.
        width (int): Number of columns in the image.
    Returns:
        rois_row (dict): Dictionary with all the metadata of the roi.
        mask (np.ndarray): Binary image of the contour of the lesion.
        maxpoint (list): Maximum coords of the bbox of the lesion.
        minpoint (list): Minimum coords of the bbox of the lesion.
    """
    mask = np.zeros((height, width))
    rois_row = {}
    for n, roi_key in enumerate(roi_dict):
        if roi_key.tag != 'key':
            continue
        if roi_key.text == 'Area':
            rois_row['area'] = float(roi_dict[n+1].text)
        elif roi_key.text == 'Center':
            rois_row['center'] = [parse_string_tuple(roi_dict[n+1].text)]
        elif roi_key.text == 'IndexInImage':
            rois_row['index'] = int(roi_dict[n+1].text)
        elif roi_key.text == 'Max':
            rois_row['max'] = int(roi_dict[n+1].text)
        elif roi_key.text == 'Mean':
            rois_row['mean'] = float(roi_dict[n+1].text)
        elif roi_key.text == 'Min':
            rois_row['min'] = int(roi_dict[n+1].text)
        elif roi_key.text == 'Name':
            rois_row['name'] = roi_dict[n+1].text
        elif roi_key.text == 'Point_mm':
            rois_row['point_mm'] = [[]]
            for point in roi_dict[n+1]:
                rois_row['point_mm'][0].append(parse_string_tuple(point.text))
        elif roi_key.text == 'Point_px':
            rois_row['point_px'] = [[]]
            for point in roi_dict[n+1]:
                rois_row['point_px'][0].append(parse_string_tuple(point.text))
            minx, maxx, miny, maxy, passed = \
                sanity_check_wz(rois_row['point_px'][0], width, height)
            mask = add_lesion_mask(mask, rois_row['point_px'][0])
            maxpoint = [maxx, maxy]
            minpoint = [minx, miny]
        elif roi_key.text == 'Total':
            rois_row['total'] = int(roi_dict[n+1].text)
        elif roi_key.text == 'Type':
            rois_row['type'] = int(roi_dict[n+1].text)
    return rois_row, mask, maxpoint, minpoint, passed


def readxml(filename: Path, height: int, width: int):
    """
    Parse the image metadata xml
    Args:
        filename (Path): Path to the xml file.
        height (int): Number of rows in the image.
        width (int): Number of columns in the image.
    Returns:
        images_df (pd.DataFrame): Containing id and number of rois.
        rois_df (pd.DataFrame): Containing all the metadata associated
            with each lesion.
        mask (np.ndarray): Binary mask of the contours of the lesions.
        maxpoint (list): Maximum coordinates for each lesion bbox.
        minpoint (list): Minimum coordinates for each lesion bbox.
    """

    rois_df = pd.DataFrame(columns=[
        'area', 'center', 'index', 'max', 'mean', 'min',
        'name', 'point_mm', 'point_px', 'total', 'type'
    ])
    images_df = pd.DataFrame(columns=['id', 'n_rois'])
    images_row = {}

    mytree = ET.parse(filename)
    root = mytree.getroot()

    id_ = filename.name.split('_')[0]
    maxpoint = []
    minpoint = []
    mask = np.zeros((height, width))

    images_row['id'] = id_
    for n0, child0 in enumerate(root[0]):
        if child0.text == 'Images':
            for n1, img_dict in enumerate(root[0][n0+1]):
                for n2, img_key in enumerate(img_dict):
                    if img_key.tag == 'key':
                        if img_key.text == 'NumberOfROIs':
                            images_row['n_rois'] = int(img_dict[n2+1].text)
                            if images_row['n_rois'] == 0:
                                break
                        if img_key.text == 'ROIs':
                            for roi_dict in img_dict[n2+1]:
                                rois_row, mask_, maxpoint_, minpoint_, passed = \
                                    parse_roi_dictionary(roi_dict, height, width)
                                mask = np.logical_and(mask, mask_)
                                maxpoint.append(maxpoint_)
                                minpoint.append(minpoint_)
                                if not passed:
                                    logging.warning(
                                        f'Case {id_}, roi {rois_row["index"]}'
                                        f' has a coordinate outside the image'
                                        f' boundaries. \n Revise file {filename}'
                                    )
                                rois_df = pd.concat([rois_df, pd.DataFrame.from_dict(rois_row)])
        images_df = pd.concat([images_df, pd.DataFrame.from_dict(images_row)])
    return images_df, rois_df, mask, maxpoint, minpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ib-path", help="path to the base directory of the database *INbreast Release 1.0*")
    parser.add_argument(
        "--masks-dir", help="path to the directory where all the masks are going to be stored")
    parser.add_argument(
        "--csvs-dir", help="path to the directory where the csvs are going to be stored")
    # TODO: add png_path passing
    args = parser.parse_args()

    base_path = Path(args.ib_path)
    csvs_path = Path(args.csvs_dir)
    dcm_folder = base_path / 'AllDICOMs'
    xml_folder = base_path / 'AllXML'
    # bbox_file = base_path / 'detectannotmass.txt'

    masks_folder = base_path / 'AllMasks'
    png_folder = base_path / 'AllPNGs'
    if not masks_folder.exists():
        masks_folder.mkdir(parents=True, exist_ok=True)
    if not png_folder.exists():
        png_folder.mkdir(parents=True, exist_ok=True)

    rois_df = pd.DataFrame(columns=[
        'area', 'center', 'index', 'max', 'mean', 'min',
        'name', 'point_mm', 'point_px', 'total', 'type'
    ])
    images_df = pd.DataFrame(columns=['id', 'n_rois'])

    k = 0
    for filename in dcm_folder.iterdir():
        id_ = filename.name.split('_')[0]
        mask_filepath = masks_folder / f'{id_}_mask.png'

        # Avoid reprocessing
        if mask_filepath.exists():
            continue

        # TODO: Check if with OpenCV or SimpleITK is faster:
        #     im = sitk.ReadImage(dcm_path)
        im = pydicom.dcmread(filename)
        im = im.pixel_array
        im = ((im.astype('double') + 0.0) / 4095.0) * 255.0
        im = im.astype('uint8')

        xml_filepath = xml_folder / f'{id_}.xml'
        if not xml_filepath.exists():
            cv2.imwrite(masks_folder/f'{id_}_mass.jpg', im)
            continue

        image_df, roi_df, mask, maxpoint, minpoint = \
            readxml(xml_filepath, im.shape[0], im.shape[1])
        images_df = pd.concat([images_df, image_df])
        rois_df = pd.concat([rois_df, roi_df])

        boundbox = np.zeros((im.shape[0], im.shape[1]))
        for j in range(np.asarray(minpoint).shape[0]):
            boundbox[minpoint(j, 1): maxpoint(j, 1), maxpoint(j, 2)] = 255
            boundbox[minpoint(j, 1): maxpoint(j, 1), minpoint(j, 2)] = 255
            boundbox[minpoint(j, 1), minpoint(j, 2): maxpoint(j, 2)] = 255
            boundbox[maxpoint(j, 1), minpoint(j, 2): maxpoint(j, 2)] = 255
        mask[boundbox == 255] = 255
        # This one may be faster but is different from the original Matlab version
        # cv2.getStructuringElement()
        mask = cv2.dilate(mask, disk(10), iterations=1)
        cv2.imwrite(masks_folder/f'{id_}_mass.png', mask)
        cv2.imwrite(png_folder/f'{id_}.png', im)
        if k == 5:
            break
        else:
            k += 1
    image_df.to_csv(csvs_path/'images_metadata.csv')
    image_df.to_csv(csvs_path/'rois_metadata.csv')

# TODO: Add bbox/rois cropping (steal Robert's code)

if __name__ == "__main__":
    main()
