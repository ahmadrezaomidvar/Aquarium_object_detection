# Aquarium_object_detection

This is a pytorch implementation of a faster RCNN network for Aquarium_object_detection.

In training, [Aquarium Object detection data by roboflow](https://www.kaggle.com/paulrohan2020/aquarium-object-detection-4817-bounding-boxes) is used.


## Dependencies
- 3.6=<Python3.6<3.9

## Setup
To make the environment:

```sh
conda env create -f environment.yml
```

Model configuration can be set in [config.yaml](./config/config.yaml)

### Create training data

dataset shall be cleaned using:

```sh
python clean_data.py
```

### Train model 

Model can be trained using:

```sh
python train.py
```


### Use trained model for prediction

Some one can use pretrained model as:

```sh
python predict.py
```


## Notes:


## License


## References