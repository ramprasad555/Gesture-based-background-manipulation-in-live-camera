import json

def merge_json_files(file1, file2, output_file):
    # Read the content from the first JSON file
    with open(file1, 'r') as f1:
        data1 = json.load(f1)

    # Read the content from the second JSON file
    with open(file2, 'r') as f2:
        data2 = json.load(f2)

    # Merge the data from both files
    merged_data = {**data1, **data2}

    # Write the merged data to the output JSON file
    with open(output_file, 'w') as output:
        json.dump(merged_data, output, indent=2)

# Replace 'file1.json', 'file2.json', and 'output.json' with your file names
merge_json_files('keypoints_all.json', 'keypoints_like.json', 'output.json')
