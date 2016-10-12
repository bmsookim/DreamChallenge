# The Digital Mammography DREAM Challenge
Repository for 2016-2017 Dream Challenge

This implements preprocessing and training of the 2016 Digital Mammography Dream Challenge.

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
