# Dependencies

* pydicom `0.9.9`
* PyYAML `3.12`


# Execution

| arg name      | arg flag | valid values               | required |
|---------------|----------|----------------------------|----------|
| `--corpus`    | `-c`     | [ `InBreast`, `dreamCh` ]  | True     |
| `--dataset`   | `-d`     | [`pilot`, `train`, `test`] | True     |
| `--format`    | `-f`     | [`class`, `robust`]        | True     |
| `--processor` | `-p`     | any Integer                | False    |

### Example
```Shell
$ python DREAM_DM_preprocessing.py -c dreamCh -d train -f class
```
means that program will use  dataset in config[data][dreamCh][pilot]

# Configuration

You can see the example of configuration in `config/preprocessing.yaml'.

```yaml
data
    #<corpus>
        #<dataset>
        metadata
            dir: #<root dir>
            images_crosswalk: images_crosswalk.tsv
            exams_metadata  : exams_metadata.tsv

preprocessing
    pipeline: [modify, feature, adjust]
    modify  :
        # [resize, flip, trim, padding, colormap(bone)]
        pipeline: [resize, flip, trim, padding, colormap]
    feature:
        # extraction  [orb, sift, surf]
        extraction  : surf
        # matchin:    [brute_force, flann]
        matching    : flann
        # alignment
        alignment   : pass
    adjust:
        rule        : pass
        algo        : pass

resultDir: <result directory>
logDir   : <log directory (info, error)>
```

# Pipeline

Preprocessing pipeline follows configuration. It means that you can easily modify pipeline with editing configuration file (`config/preprocessing.yaml`)

- read *.dcm file list and metadata     
    `DREAM_DM_preprocessing.build_metadata`
    - read *.dcm file list  
    `DREAM_DM_preprocessing.__build_image_data_from_metadata`
    - read meatadata and build subject/exam dict    
    `DREAM_DM_preprocessing.__build_exams_data_from_metadata`
- split dataset in the number of `proc_cnt`
- execute multi-processing
    - dicom file preprocessing  
    `DREAM_DM_preprocessing.preprocessing_dcm`
        - convert dicom file to `opencv` image  
        `Preprocessor.dcm2cvimg`
        - execute preprocessing pipelin
            - execute modification process  
            `Preprocessor.<method>`
            - execute image alignment process   
            `Preprocessor.<method>`
            - execute adjusting process 
            `Preprocessor.<method>`
    - write image
    - write metadata about current dicom file
- merge all metadata written by each process

# Result 

__`-m robust`__
```
/
|- trainingData (read-only)
|- preprocessedData
    |- dreamCh
        |- pilot
            |- metadata.tsv
            |- <patient_id>
                |- <exam_idx>
                    |- <laterality>
                        |- <view>.png
        |- train
            |- metadata.tsv
            |- <patient_id>
                |- <exam_idx>
                    |- <laterality>
                        |- <view>.png
```

__`-m class`__
```
/
|- trainingData (read-only)
|- preprocessedData
    |- dreamCh
        |- train
            |- metadata.tsv
            |- 0
                |- *.png
            |- 1
                |- *.png
```

# Benchmark

### Express-lane (1% dataset)
```
STDOUT: 2016-12-04 16:03:08,055 INFO   [DREAM_DM_preprocessing.py:219 -      preprocessing_single_proc()]  Proc32 Finish : size[13]   elapsed[6.71353411674]

STDOUT: 2016-12-04 16:03:08,096 INFO   [DREAM_DM_preprocessing.py:219 -      preprocessing_single_proc()]  Proc29 Finish : size[13]   elapsed[6.76222395897]

STDOUT: 2016-12-04 16:03:08,107 INFO   [DREAM_DM_preprocessing.py:219 -      preprocessing_single_proc()]  Proc36 Finish : size[13]   elapsed[6.74229621887]

STDOUT: 2016-12-04 16:03:08,120 INFO   [DREAM_DM_preprocessing.py:219 -      preprocessing_single_proc()]  Proc13 Finish : size[13]   elapsed[6.83266806602]

STDOUT: 2016-12-04 16:03:08,123 INFO   [DREAM_DM_preprocessing.py:219 -      preprocessing_single_proc()]  Proc12 Finish : size[13]   elapsed[6.83746790886]

STDOUT: 2016-12-04 16:03:08,146 INFO   [DREAM_DM_preprocessing.py:219 -      preprocessing_single_proc()]  Proc40 Finish : size[13]   elapsed[6.76750802994]

STDOUT: 2016-12-04 16:03:08,151 INFO   [DREAM_DM_preprocessing.py:219 -      preprocessing_single_proc()]  Proc2 Finish : size[13]   elapsed[6.87624406815]

STDOUT: 2016-12-04 16:03:08,167 INFO   [DREAM_DM_preprocessing.py:219 -      preprocessing_single_proc()]  Proc1 Finish : size[13]   elapsed[6.89332699776]

STDOUT: 2016-12-04 16:03:08,199 INFO   [DREAM_DM_preprocessing.py:219 -      preprocessing_single_proc()]  Proc43 Finish : size[13]   elapsed[6.81538200378]

STDOUT: 2016-12-04 16:03:08,239 INFO   [DREAM_DM_preprocessing.py:219 -      preprocessing_single_proc()]  Proc23 Finish : size[13]   elapsed[6.92792916298]

STDOUT: 2016-12-04 16:03:08,318 INFO   [DREAM_DM_preprocessing.py:219 -      preprocessing_single_proc()]  Proc18 Finish : size[13]   elapsed[7.0223069191]

STDOUT: 2016-12-04 16:03:08,386 INFO   [DREAM_DM_preprocessing.py:219 -      preprocessing_single_proc()]  Proc10 Finish : size[13]   elapsed[7.10315489769]
```
