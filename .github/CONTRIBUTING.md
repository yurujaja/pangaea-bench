## Contributing 

We welcome all forms of contributions, including but not limited to the following.

- Introduce new geospatial foundation models
- Incorporate downstream datasets
- Add new decoder heads
- Fix typo or bugs
- Add new decoders

### Workflow

1. fork and pull the latest repository
2. checkout a new branch (do not use the main branch for PRs)
3. commit your changes
4. create a PR

Note: For significant modifications or any bugs spotting, please consider opening an issue for discussion beforehand.

## Code structure 

### engines
In engines, basic modules in the training pipeline are defined including data_preprocessor, trainer and evaluator.
1. data_preprocessor selects the bands needed by an encoder and pads unavailable bands with zeros, and different augmentations.
2. trainer supports mixed precision/distributed training and print training stats and metrics in real time.
3. evaluator can be called independently and evaluate a model also in distributed way and compute per class metrics.

### datasets
1. The implementations are simplified and standardized.
2. Dataset metas are read from configs, including newly added classes (name), ignore_index, and so on.
3. To add (register) a new dataset implementation, use the decorator ```@DATASET_REGISTRY.register()```.

### foundation_models
1. Support multi-stage output that may be needed by segmentors, specified by output layers in encoder config.
2. All the encoder should work properly.
3. To add (register) a new encoder implementation, use the decorator ```@ENCODER_REGISTRY.register()```.

### segmentors
1. The UperNet implementation is based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/tree/main)
2. To add (register) a new encoder implementation, use the decorator ```@SEGMENTOR_REGISTRY.register()```.
3. So far, we have UPerNet for unitemporal semantic segmentation, UPerNetCD for change detection and MTUPerNet for multitemporal semantic segmentation
4. for multi-temporal, L-TAE and linear projection are supported

### augmentations
1. All the available augmentations are in ```data_preproessor.py```
2. To add (register) a new augmentation implementation, use the decorator ```@AUGMENTER_REGISTRY.register()```.

All the parameters can also be set in the run config file.

## Adding new features

### Adding a new geospatial foundation model
1. Inside the `foundation_models` folder:
- Add your model architecture. Including the decoder is optional, as the project focuses on evaluating pretrained model encoders for downstream tasks.
- Update `__init__.py` to include the model.

2. Inside the `configs/foundation_models` folder:
- Create a configuration file for the model:
    - Provide a `download_url` if available
    - Detail the model, including its support for temporality, the image size used, and the encoder's output dimension
    - Specify the parameters for initializing your model in `encoder_model_args`

### Adding a new downstream dataset
1. Inside the `datasets` folder:
- Add your dataset file. 
- In `__getitem__` function, , structure the output based on the modalities available in your dataset as follows:
    ```
        {
            'image': {
                'optical': optical_tensor,
                'sar' : sar_tensor,
            },
            'target': target_tensor,
            'metadata': {
                "info1": info1,
            }
        }
    ```
    - For uni-temporal dataset, shape the image tensors (C, H, W)
    - For uni-temporal dataset, shape the image tensors (C, T, H, W)

- Implement a `get_splits` function to manage dataset train/val/test splits. Use other datasets as references.
- Update `__init__.py` to include the dataset.

2. In the `configs/datasets` folder:
- Add a configuration for the dataset:
    - Provide a `download_url` if possible
    - For uni-temporal dataset, set `multi_temporal` to `False`; for multi-temporal dataset, indicate the number of time frames used, e.g., `multi_temporal: 6`
    - Include information about the dataset bands, including types and statistics
    - Provide information about the dataset classes, such as number of classes, names, ignore index, and distribution