import json

# Read the JSON file
with open('/home/chge7185/repositories/InstaFormer/helper/data_cfg/ADEMap24.json') as file:
    data = json.load(file)

# Define the translation mapping
translation_mapping = {
    '0': '0',
    '1': '1',
    '2': '2',
    '3': '14',
    '4': '14',
    '5': '3',
    '6': '4',
    '7': '11',
    '8': '11',
    '9': '5',
    '10': '6',
    '11': '7',
    '12': '6',
    '13': '8',
    '14': '11',
    '15': '9',
    '16': '6',
    '17': '10',
    '18': '6',
    '19': '6',
    '20': '12',
    '21': '13',
    '22': '14',
    '23': '15',
    # Add more mappings as needed
}

# Translate the semantic classes to new values
translated_data = {key: translation_mapping.get(str(value)) for key, value in data.items()}

data=translated_data

# Print the translated data
print(data)

with open('/home/chge7185/repositories/InstaFormer/helper/data_cfg/ADE20kClasses.json') as f:
    old_cls_info = json.load(f)
with open('/home/chge7185/repositories/InstaFormer/helper/data_cfg/16Classes.json') as f:
    cls_info = json.load(f)
# Print the mapping of labels
for c in old_cls_info:
    print("{} -> {}".format(old_cls_info[c]["name"], cls_info[str(data[c])]["name"]))

with open('/home/chge7185/repositories/InstaFormer/helper/data_cfg/ADEMap16.json', mode="w") as file:
    json.dump(translated_data, file, indent=4)
