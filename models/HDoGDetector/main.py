from marasinou_class import HDoGCalcificationDetection
from database.dataset import INBreast_Dataset
# import time
import logging
# from tqdm import tqdm


logging.basicConfig(level=logging.INFO)


def main():
    logging.info('Loading database')
    db = INBreast_Dataset(
        return_lesions_mask=True,
        level='image',
        extract_patches=False,
        normalize=None,
        n_jobs=-1,
        partitions=['train']
    )

    # Default parameters
    dog_parameters = {
        'min_sigma': 1.18,
        'max_sigma': 3.1,
        'sigma_ratio': 1.05,
        'n_scales': None,
        'dog_blob_th': 0.006,
        'dog_overlap': 1
    }

    hessian_parameters = {
        'method': 'alex',
        'hessian_threshold': 1.4,
        'hessian_th_divider': 200.
    }

    idx = 0
    for i in range(1, 500, 50):
        hessian_parameters['hessian_th_divider'] = i
        detector = HDoGCalcificationDetection(dog_parameters, hessian_parameters)
        _, _ = detector.detect(db[idx]['img'], db.df.at[idx, 'img_id'])
    detector.delete_hdog_file()


if __name__ == '__main__':
    main()
