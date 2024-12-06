import json

def merge_json_files(file1, file2, output_file):
    with open(file1, 'r') as f:
        data1 = json.load(f)
    
    with open(file2, 'r') as f:
        data2 = json.load(f)
    
    merged_data = {**data1, **data2}
    
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=4)

# Example usage
file1 = 'initData/MS_COCO/training_set/annotations/captions_train2017.json'
file2 = 'initData/MS_COCO/extra_train_2017/annotations/captions_extra_train_2017.json'
output_file = 'merged_captions.json'

merge_json_files(file1, file2, output_file)