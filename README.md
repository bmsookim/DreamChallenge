# The Digital Mammography DREAM Challenge
Repository for 2016-2017 Dream Challenge

This implements preprocessing and training of the 2016 Digital Mammography Dream Challenge.

# Content

* Preparation
* Dataset
* Preprocessing
    * Preparation
        * Dataset & Metadata
        * Configuration
    * Pipeline
        * Modules
        * Pipeline Example
    * Result
        * Directory Hierarchy
        * Benchmark
* Training
* Inference

# Preparation


# Preprocessing

## Preparation

### Dataset & Metadata

### Configuration

You can see the example of configuration in [preprocessing.yaml.sample](pre-processing/config/preprocessing.yaml.sample).

```yaml
# Dataset paths
data
    <corpus>
        <dataset>
        metadata
            dir: #<root dir>
            images_crosswalk: images_crosswalk.tsv
            exams_metadata  : exams_metadata.tsv

# result root directory path
resultDir: /preprocessedData
logDir   : /preprocessedData/log/

# file sampling
#   default | undersampling | manual (not yet)
sampling: default

# Preprocessing pipeline
#   trim
#   roi
#   crop
#   padding
#   resize
#   registration (not yet)
pipeline: 
    prev_norm   : [contrast]
    prev_roi    : [trim, flip]
    roi: True
    post_roi    : [crop, padding, resize]
channel: [gray, mass, calcification]

# pipeline modules
modules:
    contrast:
        target_mean: 700
        threshold  : 1000
        clip:
            min: 0
            max: 4000
    normalize:
        max: 255.0
    trim: null
    flip:
        target: R
        direction: H
    roi:
        # target: [mass, calcification]
        targets: [mass, calcification]
        mass:
            threshold: 0.7
        calcification:
            threshold: 0.5
        nipple:
            threshold: 0.7
    crop:
        # method: centered | roi-boundary
        method: centered
        min   : 1024
        padding: 10
    padding: null
    resize:
        # method: NEAREST | LINEAR | AREA | CUBIC | LANCZOS4
        interpolation: LINEAR
        size:
            width:  256
            height: 256
    colormap: null

```


## Pipeline

You can build the pipeline by modifying the configuration file without any code modification.

```yaml
....
sampling: default
....
pipeline: 
    prev_norm   : [contrast]
    prev_roi    : [trim, flip]
    roi: True
    post_roi    : [crop, padding, resize]
channel: [gray, mass, calcification]
...
```


### Modules

* contrast

* normalize

* trim

* flip

* roi

* crop

* padding

* resize

* colormap

### Pipeline Example

## Result

### Directory Hierarchy
__`-m robust`__
```
<Result Root>
|- trainingData (read-only)
|- preprocessedData
    |- dreamCh
        |- pilot
            |- metadata.tsv
            |- <subject_id>
                |- <exam_idx>
                    |- <laterality>
                        |- <view>.png
        |- train
            |- metadata.tsv
            |- <subject_id>
                |- <exam_idx>
                    |- <laterality>
                        |- <view>.png
```

__`-m class`__
```
<Result Root>
|- trainingData (read-only)
|- preprocessedData
    |- dreamCh
        |- train
            |- metadata.tsv
            |- 0
                |- <subject_id>_<exam_id>_<view>_<laterality>.png
            |- 1
                |- <subject_id>_<exam_id>_<view>_<laterality>.png
```

### Benchmark





## Requirements
See the [installation instructions](INSTALL.md) for a step-by-step guide.
- Install [OpenCV]()
- Install [Pydicom]()

## Directories
- Create a file named [traningData]() containing the .dcm files of mammography for training.
- Create a file named [testData]() containing the .dcm files for test phase.
- Create a file named [metadata]() containing [exams\_metadata.tsv]() and [images\_crosswalk.tsv]()

## Preprocessing
The preprocessing stage is consisted as below :
- Opening the dicom files in trainingData and extracting pixel informations from each files.
- Converting the pixel informations into a gray-scale.
- Clipping out the text section annotating the view of the certain file.
- Seperating each files into the matching directories in [preprocessedData]() according to their class.

```bash
./preprocssing.sh
```
