## Arguments

| arg name      | arg flag | valid values               | required | description                                                                                        |   |
|---------------|----------|----------------------------|----------|----------------------------------------------------------------------------------------------------|---|
| `--corpus`    | `-c`     | [ `InBreast`, `dreamCh` ]  | True     | pass the corpus name which will be used in preprocessing                                           |   |
| `--dataset`   | `-d`     | [`pilot`, `train`, `test`] | True     | pass the dataset name which will be used in preprocessing                                          |   |
| `--metadata`  | `-m`     | any String                 | False    | pass the metadata directory, if not, will use dicom files in dataset directory to build image data |   |
| `--processor` | `-p`     | any Integer                | False    | how may use processores, if not, will use `machine_cpu_cnt -1`                                     |   |

### Example
```Shell
$ python preprocessing.py -c dreamCh -d pilot -m /home/yumihwan/workspace/data/DreamChallenge_mammo/dataset/pilot_metadata/ -p 3
```


### Result 
```
Root
|- *.sh
|- *.py
|- trainingData
|- testData
|- preprocessedData
    |- pilot
        |- ...
    |- trainingData
        |- <patient_id>
            |- <exam_idx>
                |- meta.tsv
                |- <view>
                    |- <laterality>
                        |- *.png
|- testData
|- ...
```
