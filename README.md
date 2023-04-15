### 3D Segmentation Pipeline on Agricultural Point Cloud Datasets

In this repository we present experimentation for the IE 694's AI in agriculture portfolio.
This codebase contains a point cloud segmentation pipeline for _Fuji Apple_ 3D datasets as shown in Figure 1.
Further, we experiment with different model iterations to further quantitatively evaluate the
segmentation performance of different data processing alternatives chosen for model training.

<p align="center">
  <img src="pipeline-artifacts/assets/point-cloud-data-processing-pipeline.png" />
</p>
<p align="center">
<b>Figure 1:</b> The point cloud segmentation pipeline for processing large scale agriculture datasets.
</p>

## Table of Contents

* [Dataset Description of `Fuji Apple` Point Cloud Datasets](#dataset-description-of-fuji-apple-point-cloud-datasets)
* [Preprocessed `PCL` datasets Descriptions and Download Links](#preprocessed-pcl-datasets-descriptions-and-download-links)
* [`PCL` Upsampling Insights](#pcl-upsampling-insights)
* [Experimentation Summarization](#experimentation-summarization)
* [Citing the Experiment Findings and Accompanying Theoretical Document](#citing-the-experiment-findings-and-accompanying-theoretical-document)


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

#### `PCL` Upsampling Insights

The  point cloud datasets under analysis are highly undersampled and contain only sparse representations of apple point clouds.
Hence, upsampling the apple point clouds as pre-processing can help the model learn better representations for apple class objects.
Since, the complete point cloud size is very large for _PFuji-SfM_ dataset, we apply the upsampling operation on its patches only.
Whereas for the relatively smaller _Fuji-SfM_ we upsample the apple point cloud when the complete dataset is still together as a monolith.
In our upsampling technique we randomly pick an apple point cloud class point and find its nearest neighbor, after that take an average of
_{X, Y, Z, R, G, B, F1, ... , FK}_ values of the selected random point and nearest neighbor.
And, add this new point to the list of newly generated point clouds, and when this new list size reaches 80 \% existing point cloud size,
then the complete point cloud is updated with the addition of newly generated point clouds.
In below Table 1 and 2, we highlight the difference between original PCL patches, 2X apple point cloud upsampled patches, and
3X apple point cloud upsampled patches.
The point cloud plotter downsampled the points to a maximum of 40K points, therefore the apple point cloud density appears to be almost
similar for all subplots. But, with close observation we can observe that the density of the background class representations is decreasing
as the upsampling factor is increasing.

| **Original Data Patch** | **2X Upsampled Patch** | **3X Upsampled Patch** |
| --- | --- | --- |
| ![Original Patch One](pipeline-artifacts/assets/data-patch-one-org.png) | ![2X Upsampled Patch One](pipeline-artifacts/assets/data-patch-one-double.png) | ![3X Upsampled Patch One](pipeline-artifacts/assets/data-patch-one-triple.png) |
| ![Original Patch Two](pipeline-artifacts/assets/data-patch-two-org.png) | ![2X Upsampled Patch Two](pipeline-artifacts/assets/data-patch-two-double.png) | ![3X Upsampled Patch Two](pipeline-artifacts/assets/data-patch-two-triple.png) |
<p align="center">
<b>Table 1:</b> The point cloud upsampling comparison table for <em>Fuji-SfM</em> dataset's point cloud data patches.
</p>

| **Original Segmented Patch** | **2X Upsampled Patch** | **3X Upsampled Patch** |
| --- | --- | --- |
| ![Original Patch One](pipeline-artifacts/assets/seg-data-patch-one-org.png) | ![2X Upsampled Patch One](pipeline-artifacts/assets/seg-data-patch-one-double.png) | ![3X Upsampled Patch One](pipeline-artifacts/assets/seg-data-patch-one-triple.png) |
| ![Original Patch Two](pipeline-artifacts/assets/seg-data-patch-two-org.png) | ![2X Upsampled Patch Two](pipeline-artifacts/assets/seg-data-patch-two-double.png) | ![3X Upsampled Patch Two](pipeline-artifacts/assets/seg-data-patch-two-triple.png) |
<p align="center">
<b>Table 2:</b> The point cloud upsampling comparison table for <em>Fuji-SfM</em> dataset's segmented point cloud data patches.
</p>

#### Experimentation Summarization

In our experiments we trained four different models as described in the above experimentation methodology section.
All the trained models failed to converge with different dataset iterations and different hyperparameters, like
changed learning rate, batch size etc.
This majorly can be attributed to the complexity of segmentation tasks where countless complex objects like, grass,
branches, trunks, and soil textures are grouped into background class. 
It becomes very hard to differentiate between different objects and classify the apple class point cloud data points.
Therefore, a point cloud dataset with more extensive segmentation annotations can help in building a segmentation model
that correctly segments out Fuji apples directly.

Further, technically the _Fuji-SfM_ dataset provides only bounding box cubical annotations which were translated to
segment apple point clouds for ground truth preparation.
These cubical bounding boxes systematically might have introduced leaf point clouds in these ground truth annotations.
Second, this dataset is rather sparse and does not provide an extensive point cloud for effective model training, 
and might require extensive data augmentation for building robust segmentation models.

And, for the _PFuji-Size_ dataset the documentation of the dataset is rather a bit ambiguous, and hard to follow up with
_*.LAZ_ format point cloud files.
In this dataset, the orchard point clouds and apple annotation files are provided separately for analysis but nowhere it
is explicitly mentioned whether the orchard point cloud files contain the apple annotations within it or not.
We assumed, that these annotations are not present in the shared orchard point clouds, and evidently the models have not
converged or learnt any valuable representations.
So, it might be possible that apple annotations are present in the _*.LAZ_ point cloud files of the orchard dataset.

**In Progress Work Note:** Currently, the _PFuji-Size_ dataset debugging, and more extensive README.md documentation
is in progress.
Additionally, we are also testing data augmentation techniques for the _Fuji-SfM_ dataset to measure any performance gains.
If some data preprocessing flaw is detected we will update the code repository accordingly, and share
the updated notebooks and model checkpoints.

#### Citing the Experiment Findings and Accompanying Theoretical Document

If you find the experimentation work interesting and useful for your work, please consider citing it with:

```
@misc{fuji-point-cloud-analysis,
  author = {Rana, Ashish},
  title = {Exploring Scope of Applying Deep Learning Techniques on 3D Agriculture Data},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/arana-initiatives/agro-point-cloud-seg-pipeline}},
}
```
