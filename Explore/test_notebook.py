
import json

nb_file = "Processing_AIS_Logs.ipynb"

with open(nb_file, 'r') as fs:
    nb= json.load(fs)

for cell in nb['cells']:
    print(cell['cell_type'])
    if cell['cell_type'] == 'code':
        source_code = cell['source']
        print("##########################################################")
        print("##########################################################")
        print("##########################################################")
        for line in source_code:
            print(line, end='')
