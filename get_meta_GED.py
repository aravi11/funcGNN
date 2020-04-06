import os
import glob
import json

avg_train_GED = 0
avg_test_GED = 0

train_ged_list = []

train_GED_dict = {}
test_GED_dict = {}

json_training_dir_name = "./dataset/train"
json_testing_dir_name = "./dataset/test"

json_pattern = os.path.join(json_training_dir_name,'*.json')
file_list = glob.glob(json_pattern)

for json_file in file_list:
    with open(json_file) as file:
        data = json.load(file)
    value = int(data['ged'])
    train_ged_list.append(value)
   
    if value in train_GED_dict:
        train_GED_dict[value] = int(train_GED_dict[value]) + 1
    else:
        train_GED_dict[value] = 1
    file.close()
    avg_train_GED += value

avg_train_GED = avg_train_GED/len(file_list)
print(avg_train_GED)
#print(train_GED_dict)
with open('./train_ged_distribution.txt', 'w') as ged_dist:
    ged_dist.write(str(train_ged_list))
ged_dist.close()
