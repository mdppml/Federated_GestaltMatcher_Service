# Federated GestaltMatcher Service
This repository contains the source code of Federated GestaltMatcher Service, enabling the federated training 
of ensemble feature extractor and privacy preserving syndrome inference and discovery introduced by 
[Hustinx et al. (2023)](https://arxiv.org/abs/2211.06764). This repo is partially based on the
public repository of this study, which can be accessed on their [Github repo](https://github.com/igsb/GestaltMatcher-Arc).

In order to reproduce the results access must be requested to the GestaltMatcher DataBase (GMDB).
That can be done following this link (https://db.gestaltmatcher.org/documents) if you're affiliated with a 
medical facility or faculty.

For more information about the setup of the environment and preprocessing of the data, please refer to [GestaltMatcher-Arc repo](https://github.com/igsb/GestaltMatcher-Arc).

## Environment
We conducted our experiments on a server running Slurm Workload Manager.
Even though we included the recipe of the container that we created, one can
run the experiments without Slurm Workload Manager by simply running the 
corresponding scripts on terminal(s).

We use Python 3.10 and the following packages listed in requirements.txt. To 
setup the environment, you can run the following bash code snippet:
```
python3 -m venv env_gm
source env_gm/Scripts/activate
pip install -r requirements.txt
```
Briefly, the required packages are listed as follows:
```
numpy==1.21.5
pandas==1.4.3
pytorch==1.12.0
torchvision==0.13.0
tensorboard==2.16.2
opencv==4.6.0
scikit-image==0.21.0
scikit-learn==1.3.2
scipy==1.10.1
onnx2torch==1.4.1
albumentations==1.2.1
```


## Data preparation from [GestaltMatcher-Arc Repo](https://github.com/igsb/GestaltMatcher-Arc)
### Dataset
The data should be stored in `../data/GestaltMatcherDB/<version>`, it can be downloaded from http://gestaltmatcher.org 
on request. \
In our experiments, we use version 1.0.3. Please download the following two files from GMDB website:
* GMDB metadata
* GMDB_original_images_v1.0.3.tar.gz

```
cd ../data/GestaltMatcherDB
tar -xzvf GMDB_original_images_v1.0.3.tar.gz
mv GMDB_original_images_v1.0.3 gmdb_images
tar -xzvf GMDB_metadata.tar.gz
mv gmdb_metadata/* .
```

Make sure your final data structure looks as follows: \
`..\data\GestaltMatcherDB\<version>`\
`...\gmdb_images`\
`...\gmdb_metadata`,\
where `<version>` is your version of GMDB. 

### Crop and align faces
In order to get the aligned images, you have to run the `detect_pipe.py` and `align_pipe.py` from 
https://github.com/AlexanderHustinx/GestaltEngine-FaceCropper. \
More details are in the README of that repo. \
Most importantly the face cropper requires the model-weights "Resnet50_Final.pth". Remember to download them from 
[Google Docs](https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1) with pw: fstq

The face cropper requires the model-weights "Resnet50_Final.pth". Remember to download them from the repository 
mentioned above.\
If you don't have GPU, please use `--cpu` to run on cpu mode.

FaceCropper command to get relevant coordinates of faces from data directory:
```
python detect_pipe.py --images_dir ../data/GestaltMatcherDB/<version>/gmdb_images/ --save_dir ../data/GestaltMatcherDB/<version>/gmdb_rot/ --result_type coords
```

FaceCropper command to align all faces based on the coordinates according to the ArcFace alignment used by insightface:
```
python align_pipe.py --images_dir ../data/GestaltMatcherDB/<version>/gmdb_rot/ --save_dir ../data/GestaltMatcherDB/<version>/gmdb_align/ --coords_file ../data/GestaltMatcherDB/<version>/gmdb_rot/face_coords.csv
```
Note: the alignment will require the `scikit-image` package.\
Make sure to replace the `<version>` in the paths with your GMDB version; highest version at the time of writing is v1.0.3

## Federated Global Ensemble Feature Extractor Model Training and Generating Latent Representations
The ensemble model contains three models, which are fine-tuned (1) ResNet-50 mix model and (2) ResNet-100 model, and
(3) the original ResNet-100 model. Therefore, we require to access the pre-trained versions of these models, which are
named as `glint360k_r50.onnx` and `glint360k_r100.onnx`, respectively. You can downloaded here:
https://github.com/deepinsight/insightface/tree/master/model_zoo

In our experiments, we store them in `/saved_models`. However, you can store them wherever you want and provide that
path to the experiment scripts.

To run the federated training of the global ensemble feature extractor model, you can run the following script:
```
./training.sh -s <session_id> -a <aggregation_method> -e <num_of_epochs> -f <aggregation_interval> -n <num_of_parties> -w <num_of_workers> -l <distribution_prefix> -d <distribution_number>
```
where
```
s: the session id to distinguish runs from each other
a: the aggregation method (currently 'mean' is populated and tested)
f: the aggregation interval, indicating after how many epochs the local models will be aggregated
n: the number of silos in the system (we created the data of 4, 8, and 16 silos for our experiments)
w: the number of workers to be used to load data
l: the distribution type prefix (near-uniform: subject-wise_complete_uniform_random_ -- non-overlapping: non-intersecting_label_distribution_)
d: the label distribution number (1-5 for near-uniform and 1 for non-overlapping
```
This information can also be obtained by running ```./training.sh -h```. Before running `training.sh`,
the paths specified in `client.sh` and `aggregator.sh` scripts run by `training.sh` needs to be adjusted. 
More specifically, the following paths/information need to be specified:
```
encoding_dir: the directory where the encodings (latent representations) will be saved
data_dir: the directory where the data resides
weight_dir: the directory storing the weights of the pre-trained networks mentioned earlier
model_dir: the directory where the final models will be saved
lookup_table_save_path: the directory where the lookup table will be saved
federated_metadata_path: the additional path info on top of data_dir storing the files of patient ids that belong to each silo
```

To give a sample running script, we provide the script to run the second near-uniform distribution experiment for 
50 epochs with 4 silos, mean aggregation, the aggregation interval of 5 epochs below:
```
./training.sh -s 36 -a mean -e 50 -f 5 -n 4 -w 4 -l subject-wise_complete_uniform_random_ -d 2
```
This script will create 5 processes in total by running `client.sh` and `aggregator.sh`. 
Four of those processes are for the clients and one for the aggregator. Each process will be 
submitted as a separate job to Slurm Workload Manager. One can run the
corresponding scripts initiating clients and aggregator on the terminal if Slurm Workload Manager is not
available.

Each client trains its local model using its corresponding _frequent_ patients. After _f_ epochs, the local models
are aggregated by the aggregator and the aggregated model sent back to the clients. This process continues until _e_
epochs. This process is performed to fine-tune both `glint360k_r50.onnx` and `glint360k_r100.onnx`, respectively.
Once the global ensemble feature extractor model is obtained, the clients compute the latent representations of their
_frequent_ and _rare_ patients' images, which then saved to `/encodings` folder.

In our experiments, we used the default seed, which is `11`. You can specify a different
seed using the argument `--seed`.

Training a model without GPU has not been tested.

### Syndrome Inference
To evaluate the performance of the federated GestaltMatcher syndrome inference, the latent representations of images
of silos computed using the global feature extractor model will be utilized. The following script runs the kernel matrix
computation framework to calculate the cosine distance between all pairwise latent representations distributed across
multiple silos. The evaluation of the federated GestaltMatcher service is then performed over the test samples, which 
are also distributed across multiple silos using Top-k evaluation metric. 

```
./inference.sh -s <session_id> -n <num_of_parties> -l <distribution_prefix> -d <distribution_number>
```

A sample running script in parallel to the above `training.sh` script is as follows:
```
./inference.sh -s 36 -n 4 -l subject-wise_complete_uniform_random_ -d 2
```

This will output of the evaluation results in a similar following format:
```
===========================================================
---------   test: Frequent, gallery: Frequent    ----------
|Test set     |Gallery |Test  |Top-1 |Top-5 |Top-10|Top-30|
|GMDB-frequent|5759    |593   |47.06 |70.29 |77.47 |86.69 |
---------       test: Rare, gallery: Rare        ----------
|Test set     |Gallery |Test  |Top-1 |Top-5 |Top-10|Top-30|
|GMDB-rare    |792.7   |312.3 |36.44 |52.82 |61.22 |75.33 |
---------   test: Frequent, gallery: Frequent+Rare   ----------
|Test set     |Gallery |Test  |Top-1 |Top-5 |Top-10|Top-30|
|GMDB-rare    |6551.7  |593 |46.34 |67.93 |74.96 |86.03 |
---------   test: Rare, gallery: Frequent+Rare   ----------
|Test set     |Gallery |Test  |Top-1 |Top-5 |Top-10|Top-30|
|GMDB-rare    |6551.7  |312.3 |23.45 |38.53 |44.33 |57.40 |
Evaluation is completed! Time to run: 544 seconds
```

## Overall Structure
To give you a better perspective on the overall folder structure, we include the structure that we set up:
```
📦 Federated GestaltMatcher Service
├── 📂 data
│   ├── 📂 GestaltMatcherDB
│   │   ├── 📂 v1.0.3
│   │   │   ├── 📂 gmdb_align
│   │   │   ├── 📂 gmdb_images
│   │   │   ├── 📂 gmdb_metadata
│   │   │   │   ├── 📂 federated_metadata
│   │   │   ├── 📂 gmdb_rot 
├── 📂 models
├── 📂 saved_models
├── 📂 lookup_table
├── 📂 encodings
├── 📂 lib
│   ├── 📂 datasets
│   │   ├── 🐍 gestalt_matcher_dataset.py
│   │   ├── 🐍 utils.py
│   ├── 📂 models
│   │   ├── 🐍 my_arcface.py
│   │   ├── 🐍 utils.py
│   ├── 🐍 fed_utils.py
│   ├── 🐍 utils.py
├── 📂 job_submissions
│   ├── 📜 training.sh
│   ├── 📜 client.sh
│   ├── 📜 aggregator.sh
│   ├── 📜 inference.sh
```

## Contact
Ali Burak Ünal

Email: ali-burak.uenal@uni-tuebingen.de

## License
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc/4.0/)
