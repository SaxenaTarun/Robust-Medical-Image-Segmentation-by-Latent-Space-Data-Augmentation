# Robust-Medical-Image-Segmentation-by-Latent-Space-Data-Augmentation

Tarun Saxena & Anson Antony</br>
OS: macOS Sonoma 14.1.1


## How to run:
For Data:
- Download Datasets:
  - [ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) for training and intra-domain testing
  - [M&Ms dataset](https://www.ub.edu/mnms/) for cross-domain testing
  - [ACDC-C dataset](https://drive.google.com/file/d/1QEpe00AaUzrRPFCSNuOsoF7KHwYG5_oB/view?usp=sharing) for robustness test. We use [TorchIO](https://torchio.readthedocs.io/) to generate corrupted ACDC test data [sample code](medseg/dataset_loader/generate_artefacted_data.py).
- Data preprocessing[[example code](medseg/dataset_loader/acdc_preprocess.py)]: 
  - Intensity normalization: image intensities are normalized to (0,1)
  - Image resampling: resample images to have a uniform pixel spacing [1.36719, 1.36719]

For Training:
