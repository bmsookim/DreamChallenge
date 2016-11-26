## Arguments

| arg name      | arg flag | valid values               | required |
|---------------|----------|----------------------------|----------|
| `--corpus`    | `-c`     | [ `InBreast`, `dreamCh` ]  | True     |
| `--dataset`   | `-d`     | [`pilot`, `train`, `test`] | True     |
| `--processor` | `-p`     | any Integer                | False    |

### Example
```Shell
$ python preprocessing.py -c dreamCh -d pilot  -p 3
```

Determine the dataset path from configuration based on arguments

__Configuration__ (`./config/preprocessing.yaml)
```yaml
data:
  dreamCh:
    pilot: /pilot
    train: /trainingData
    test:  /testData
    metadata:
      dir: /metadata
      exams_metadata: exams_metadata.tsv
      images_crosswalk: images_crosswalk.tsv
    ...

resultDir: /preprocessedData
...
```
Example execution command means that program will use  dataset in config[data][dreamCh][pilot]


### Result 
```
/
|- trainingData (read-only)
|- testData     (read-only)
|- preprocessedData
    |- dreamCh
        |- pilot
            |- metadata.tsv
            |- <patient_id>
                |- <exam_idx>
                    |- <laterality>
                        |- <view.png
        |- trainingData
            |- metadata.tsv
            |- <patient_id>
                |- <exam_idx>
                    |- <laterality>
                        |- <view.png
```
