'''
Script to convert the data (all data together) (which is output
from data_prep notebook) into direct usable form for training

It sequentially collects readings that have the same label.
Once, different label is encountered, it separates the previous
readings list into parts depending on the reqd_len (required
length of each example). Then, if too much padding is required or
number of segments is more than 5 (since that would put most of the
padding in the first and last segments (which is not good)),
then excess readings are discarded to make the length of the
readings a multiple of reqd_len. The padding is calculated as
the mean of all those readings which have been collected.
Then, padding is added to the readings (which makes length of
readings a multiple of reqd_len) and in case padding is not
required, it is automatically not added as the respective parameter
is 0 (zero). Then, those padded readings are written to a new
csv file. And the process continues.

'''

import os
import sys
import csv
import numpy as np

# constant length of each example
reqd_len = 50

path = '../data/cleaned_new3.csv'

with open(path) as f :
    reader = csv.reader(f)
    # empty list to hold readings from one example
    readings = list()

    time = float(0)
    label = ''
    for row in reader :
        t = float(row[0])
        # if reading is to be continued i.e. both timestamp increases and label stays the same
        if t >= time and row[1] == label :
            readings.append(row[2 : ])
            annotation = row[1]
            time = float(row[0])

        # if timestamp value reduces, it means start of new example
        # also, if label changes, it means start of new example
        else :
            # we need to integer divide the number of readings by `reqd_len`
            # and then segment the data into those many examples
            num_data = len(readings) // reqd_len
            # if less than required length, discard the readings
            if num_data == 0 and (reqd_len - len(readings) > 10) :
                print('readings are less than ', reqd_len, 'by ', reqd_len - len(readings))
                # prepare for taking next block of data
                readings = list()
                readings.append(row[2 : ])
                label = row[1]
                time = float(row[0])
                continue

            # calculating the amount of padding required
            length = len(readings)
            k = 0
            pad_length = (reqd_len * (num_data + 1) - length) // 2
            # if too much padding is required, discard the excess readings
            # or if more than 5 consecutive segments are there, then discard
            # excess readings, since the 1st and last will have most of the
            # padding (due to the way code is written in this script)
            if pad_length > 5 * (num_data + 1) or num_data > 4:
                readings = readings[ : num_data * reqd_len]
                pad_length = 0
            else :
                # if padding is to be added it means one example more
                num_data = num_data + 1

            length = len(readings)
            # in case unsymmetrical padding is needed
            if ((pad_length * 2) + length) < reqd_len * num_data :
                k = (reqd_len * num_data) - ((pad_length * 2) + length)

            print(len(readings) + (pad_length * 2) + k)

            # now calculating the padding values

            # converting accelerometer readings to float and in NumPy array
            readings_ = np.array(readings, dtype = float)

            padding_ = (np.mean(readings_, axis = 0)).tolist()
            # adding the annotation to the padding
            padding_.append(annotation)

            # writing the padding and data to csv
            with open('../data/new_padded_data.csv', 'a') as wf :
                wrf = csv.writer(wf)
                for i in range(pad_length) :
                    wrf.writerow(padding_)
                for reading in readings :
                    reading_ = reading
                    # add annotation to each row since we had initially removed it
                    reading_.append(annotation)
                    wrf.writerow(reading_)
                for i in range(pad_length + k) :
                    wrf.writerow(padding_)

            label = row[1]
            time = float(row[0])
            readings = list()
            readings.append(row[2 : ])