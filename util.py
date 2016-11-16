# default packages
import os

import logging
import logging.config

# installed packages
import yaml
import dicom


"""
File system utils
"""
def mkdir(dir_path):
    sub_path = os.path.dirname(dir_path)
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)

"""
Logger
"""
def build_logger(default_path='config/logging.yaml',
        default_level = logging.DEBUG,
        env_key='LOG_CFG'):
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        for handler in config['handlers'].keys():
            if 'filename' in config['handlers'][handler]:
                mkdir(config['handlers'][handler]['filename'])
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=devault_level)

    return logging

"""
dicom
"""
def load_dcm(dcm_filePath):
    return dicom.read_file(dcm_filePath)
