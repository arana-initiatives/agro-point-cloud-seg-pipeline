### 3D Segmentation Pipeline on Agricultural Point Cloud Datasets

In this repository we present experimentation for the IE 694's AI in agriculture portfolio.
This codebase contains a point cloud segmentation pipeline for _Fuji Apple_ 3D datasets.
Further, we experiment with different model iterations to further quantitatively evaluate the
segmentation performance of different data processing alternatives chosen for model training.

#### Dataset Description of `Fuji Apple` Point Cloud Datasets

* [`Fuji-SfM dataset`](https://www.grap.udl.cat/en/publications/fuji-sfm-dataset/): 3D point cloud
of scanned scene with annotation of apple locations for 11 Fuji apple trees where the 3D model
is generated using SfM _(Structure from Motion)_ technique.
* [`PFuji-Size dataset`](https://www.grap.udl.cat/en/publications/apple_size_estimation_SfM/): It consists of 3D point
cloud of 615 Fuji apples scanned in-field conditions for three Fuji apple trees using SfM and
MVS _(Multi-View-Stereo)_ techniques.

#### Preprocessed `PCL` datasets Descriptions and Download Links

* [`Fuji-SfM data`](https://drive.google.com/file/d/1LHL5gp7agQyTJgodyVzjFw7qUilYZMcM/view?usp=share_link): 
Cropped dataset based on the apple locations with apple PCL KKN based upsampling upto five times the original PCL count.

* [`PFuji-Size-2018-Orchard data`](https://drive.google.com/file/d/19LEgF3_Q5MyxDm9_Ci4kg1VInfCS-oD7/view?usp=sharing): 
Cropped dataset with combined east and west 2018 orchard PCL data with no upsampling, but individual apple patches needs
to be upsampled by 2X factor during the patch generation phase.

* [`PFuji-Size-2020-Orchard data`](https://drive.google.com/file/d/10rwpTwny6eRYvgZzP5zJ6xBornBTCMQD/view?usp=share_link): 
Cropped dataset with combined east and west 2020 orchard PCL data with no upsampling, but individual apple patches needs
to be upsampled by 2X factor during the patch generation phase. 
