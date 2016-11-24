import os
import time, datetime
import argparse
import numpy as np
import network
from network import *
import csv

def run(clf):
    records = []
    save_epoch = 0
    save_dir = "../models/" + cf.dataset# + ("/%s" % clf.__class__.__name__)

    start_time = time.time()
    save_epoch = 0

    for epoch in range(save_epoch, cf.epochs):
        print "Epoch #%d" % (epoch + 1)
        epoch_time = time.time()
        clf.fit()
        duration = time.time() - start_time
        time_per = time.time() - epoch_time
        print("Elpased time : %.3f sec\n" %time_per)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_name', type=str, default=cf.model, choices=dir(network))
    args = parser.parse_args()
    run(eval("%s()" % args.class_name))
                                                  
