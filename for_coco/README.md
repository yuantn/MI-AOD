# Data Preparation of MS COCO

Language: [简体中文](./README_cn.md) | English

There are 5 python files in this folder, which are used for converting MS COCO JSON annotation format to PASCAL VOC XML annotation format (refer to [KapilM26/coco2VOC](https://github.com/KapilM26/coco2VOC)).

The following instructions are for data preparation of MS COCO. Please ensure you have finished the installation in [installation.md](../docs/installation.md).

1. Please create a file directory tree as below, where the images and annotations of MS COCO ( *train* and *val* ) can be downloaded from here:

    Images ( *train* ): http://images.cocodataset.org/zips/train2017.zip
    
    Images ( *val* ): http://images.cocodataset.org/zips/val2017.zip
    
    Annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip

    ```
    ├── coco (download from official website)
    │   ├── annotations
    │   │   ├── instances_train2017.json
    │   │   ├── instances_val2017.json
    │   ├── images (combination of train2017 and val2017)
    │   ├── train2017
    │   ├── val2017
    ├── coco2voc (manually create)
    │   ├── Annotations (empty)
    │   ├── ImageSets
    │   │   ├── Main (empty)
    │   ├── JPEGImages (symbol link -> coco/images)
    ```
    
    You may also use the following commands directly:
    
    ```bash
    cd $YOUR_DATASET_PATH
    mkdir coco
    cd coco
    wget http://images.cocodataset.org/zips/train2017.zip  # it would cost much time
    unzip train2017.zip  # it would cost much time
    wget http://images.cocodataset.org/zips/val2017.zip
    unzip val2017.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip annotations_trainval2017.zip
    mkdir images
    cp train2017/* images/
    cp val2017/* images/
    cd ../
    mkdir coco2voc
    mkdir cocovoc/Annotations
    mkdir coco2voc/ImageSets
    mkdir coco2voc/ImageSets/Main
    ln -s coco/images coco2voc/JPEGImages
    ```
    
2. Please modify the corresponding dataset directory in `MIAOD.py` in this folder, they are located in:
    
    ```python
    Line 1: data_root_coco='$YOUR_DATASET_PATH/coco/'
    Line 2: data_root_voc='$YOUR_DATASET_PATH/coco2voc/'
    ```
    
    Please change the `$YOUR_DATASET_PATH`s above to your actual dataset directory (i.e., the directory where you create the `coco` folder).

    And please use the absolute path (i.e., start with `/`) but not a relative path (i.e., start with `./` or `../`).
    
3. Please copy these 4 files to the following directory and replace the corresponding original files:

    - active_datasets.py -> mmdet/utils/active_datasets.py
    - MIAOD.py -> configs/MIAOD.py
    - train.py -> tools/train.py
    - voc.py -> mmdet/datasets/voc.py
   
   The commands are:
   
   ```bash
   cp active_datasets.py ../mmdet/utils/active_datasets.py
   cp MIAOD.py ../configs/MIAOD.py
   cp train.py ../tools/train.py
   cp voc.py ../mmdet/datasets/voc.py
   ```

4. Please install `pascal_voc_writer` package to write PASCAL VOC XML annotation as below.

    ```bash
    pip install pascal_voc_writer
    ```
    
5. Please convert the MS COCO JSON annotation format to PASCAL VOC XML annotation format as below.

    ```bash
    python coco2voc.py \
        --ann_file $YOUR_DATASET_PATH/coco/annotations/instances_train2017.json \
        --output_dir $YOUR_DATASET_PATH/coco2voc/Annotations/
    ```
    
   Please change the `$YOUR_DATASET_PATH`s above to your actual dataset directory (i.e., the directory where you create the `coco` folder).

Now you have finished the data preparation on MS COCO. Please follow [here](../README.md#train-and-test) for remaining training and test steps, and please replace `--eval mAP` with `--eval bbox` in the test step for MS COCO.
