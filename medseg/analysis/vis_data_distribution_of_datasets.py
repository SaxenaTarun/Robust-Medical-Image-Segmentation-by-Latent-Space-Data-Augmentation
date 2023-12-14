# Name: Tarun Saxena & Anson Antony
# CS 7180 Advanced Perception
# Date: 7 December, 2023

# vis_data_distribution_of_datasets.py:- loads and compares intensity distributions of cardiac images from two datasets (ACDC and M&Ms) using medical image segmentation transformations.


from medseg.dataset_loader.transform import Transformations  #
from medseg.dataset_loader.cardiac_ACDC_dataset import CardiacACDCDataset
from medseg.dataset_loader.cardiac_MM_dataset import Cardiac_MM_Dataset
from torch import normal

from medseg.analysis.vis_intensity_distribution import plt_intensity_distribution
from medseg.dataset_loader.dataset_utils import get_all_image_array_from_datastet
from medseg.common_utils.basic_operations import check_dir

if __name__ == '__main__':
    image_size = (192, 192, 1)
    pad_size = (192, 192, 1)
    crop_size = (192, 192, 1)
    new_spacing = [1.36719, 1.36719, -1]
    tr = Transformations(data_aug_policy_name='no_aug', pad_size=pad_size, crop_size=crop_size).get_transformation()

    ACDC_ES = CardiacACDCDataset(
        transform=tr['train'], data_setting_name='standard', subset_name='ES', split='train', new_spacing=new_spacing, normalize=True)
    MM_ES = Cardiac_MM_Dataset(transform=tr['train'], new_spacing=new_spacing, normalize=True)

    ACDC_ES_array = get_all_image_array_from_datastet(ACDC_ES)
    MM_ES_array = get_all_image_array_from_datastet(MM_ES)
    check_dir('./output', create=True)
    plt_intensity_distribution([ACDC_ES_array, MM_ES_array], labels=['ACDC', 'M&Ms'],
                               title='ACDC vs M&Ms', save_path='./output/ACDCvsMM_intensity.png')
