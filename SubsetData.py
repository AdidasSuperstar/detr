import pandas as pd
import shutil
import os
import numpy as np
import fnmatch

my_subset_csv = pd.read_csv('C:\\Users\\Eva.Locusteanu\\PycharmProjects\\detr\\Point_five_Percent_MiniTrainingData.csv')

print(my_subset_csv.head())

# get image ids from 1% subset
list_names_subset = my_subset_csv['image_id'].tolist()
print(list_names_subset)
print(len(list_names_subset))
# initializing append_str
append_str = '.jpg'

# Append .jpg suffix to strings in list
my_list = [sub + append_str for sub in list_names_subset]

dirs_list = [('C:\\Users\\Eva.Locusteanu\\PycharmProjects\\detr\\models\\train\\', 'C:\\Users\\Eva.Locusteanu\\PycharmProjects\\detr\\models\\train_subset_2\\')]

for img in my_list:
    for source_folder, destination_folder in dirs_list:
        shutil.copy(source_folder+img, destination_folder+img)