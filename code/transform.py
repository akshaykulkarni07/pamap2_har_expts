'''
This script is to convert the data from original dat format to csv
for future ease of use.
'''

import csv
import os

for filename in os.listdir('../PAMAP2_Dataset/Protocol/'):
    datContent = [i.strip().split() for i in open(os.path.join('../PAMAP2_Dataset/Protocol/', filename)).readlines()]
    filename_ = filename[ : -4] + '.csv'
    with open(os.path.join('../data/', filename_), 'w') as f :
        writer = csv.writer(f)
        writer.writerows(datContent)
