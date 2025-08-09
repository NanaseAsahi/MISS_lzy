import pandas 
import json
from pathlib import Path


path = Path('/home/lzy/data/MISS2/result')
json_files = list(path.glob('*.json'))

# print(json_files)


for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)

    file_name = str(json_file).split('.')[0].split('/')[-1]
    stats_json = {}
    # print(type(data))
    # print(data)

    for item in data:
        dataset = item['data']
        missingrate = item['missingrate']
        type = item['type']

        stats_json[dataset] = 

# print(json_files)
# print(str(json_files[0]).split('.')[0].split('/')[-1])







