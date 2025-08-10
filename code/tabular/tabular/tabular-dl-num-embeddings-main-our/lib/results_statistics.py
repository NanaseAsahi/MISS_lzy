import pandas 
import json
from pathlib import Path
import numpy as np


path = Path('/home/lzy/data/MISS2/result')
json_files = list(path.glob('*.json'))
json_files = [f for f in json_files if 'stats' not in str(f)]  # 不需要对stats文件进行处理

# print(json_files)


for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)

    file_name = str(json_file).split('.')[0].split('/')[-1]
    stats_json = {}
    # print(type(data))
    # print(data)

    for dataset in ['HI', 'gas', 'News', 'temperature']:
        stats_json[dataset] = {}
        for missingrate in [0.1, 0.3, 0.5, 0.7, 0.9]:
            stats_json[dataset][missingrate] = {}
            for type in ['mcar_', 'mar_p_', 'mnar_p_']:
                stats_json[dataset][missingrate][type] = {}
                stats_json[dataset][missingrate][type]['all_auroc'] = []
                stats_json[dataset][missingrate][type]['all_accuracy'] = []
                stats_json[dataset][missingrate][type]['all_time'] = []
                stats_json[dataset][missingrate][type]['stats_auroc'] = 0.0
                stats_json[dataset][missingrate][type]['stats_accuracy'] = 0.0
                stats_json[dataset][missingrate][type]['stats_time'] = 0.0

                


    for item in data:
        dataset = item['data']
        missingrate = item['missingrate']
        type = item['type']

        time = item['time']
        auroc = item['best_test_auroc']
        accuracy = item['best_test_accuracy']
        stats_json[dataset][missingrate][type]['all_auroc'].append(auroc)
        stats_json[dataset][missingrate][type]['all_accuracy'].append(accuracy)
        stats_json[dataset][missingrate][type]['all_time'].append(time)

    for dataset in stats_json:
        for missingrate in stats_json[dataset]:
            for type in stats_json[dataset][missingrate]:
                auroc_list = stats_json[dataset][missingrate][type]['all_auroc']
                accuracy_list = stats_json[dataset][missingrate][type]['all_accuracy']
                time_list = stats_json[dataset][missingrate][type]['all_time']
                    
                stats_json[dataset][missingrate][type]['stats_auroc'] = np.mean(auroc_list) if auroc_list else 0.0
                stats_json[dataset][missingrate][type]['stats_accuracy'] = np.mean(accuracy_list) if accuracy_list else 0.0
                stats_json[dataset][missingrate][type]['stats_time'] = np.mean(time_list) if time_list else 0.0
            
    output_file = path / f'{file_name}_stats.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats_json, f, indent=4, ensure_ascii=False)

    print('Success!')
        


# print(json_files)
# print(str(json_files[0]).split('.')[0].split('/')[-1])







