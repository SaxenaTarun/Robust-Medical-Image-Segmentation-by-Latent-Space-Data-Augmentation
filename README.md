# Robust-Medical-Image-Segmentation-by-Latent-Space-Data-Augmentation

Tarun Saxena & Anson Antony</br>
OS: macOS Sonoma 14.1.1</br>
Project Presentation (14 min): https://drive.google.com/file/d/1bXmT4x8-8ELf_BHCrT7dJPaI4mbLSG0O/view?usp=drive_link 

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
To train the networks using the ACDC training set with specific configurations, use the following command:

CUDA_VISIBLE_DEVICES=0 python medseg/train_adv_supervised_segmentation_triplet.py --json_config_path "./config/ACDC/cooperative_training.json" --cval 0 --data_setting 10 --log --seed 40
Here's an explanation of the command options:

json_config_path: Path to the configuration file. You can choose cooperative training or standard training by specifying the corresponding config file path. For cooperative training (with latent space domain adaptation), use config/ACDC/cooperative_training.json. For standard training (without latent space domain adaptation), use config/ACDC/standard_training.json.

cval: Integer seed for specifying a particular random set of training subjects for multiple runs. The default is 0.

data_setting: Integer/float/string, specifying the number of subjects (n as int) or percentage (n% as float) to use for training. For example, --data_setting 10 uses 10 training subjects (limited training set). For standard training with 70 subjects (large training set), you can use --data_setting standard. Check medseg/dataset_loader/ACDC_few_shot_cv_settings.py for reference.

seed: Integer, controlling randomness during training.

log: Boolean, used for visualizing the training/validation curves. If set to true, open another terminal and run tensorboard --logdir ./saved/ --port 6006 to launch TensorBoard for visualization.

For cross-domain testing, load the trained model from a saved checkpoint and perform tests on a set of test sets. Refer to medseg/test_ACDC_triplet_segmentation.py for reference.
