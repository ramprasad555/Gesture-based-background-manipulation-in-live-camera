import json
import random

def pick_and_merge_objects(file_path, output_file, num_objects=100):
    # Read the content from the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Filter objects with "label" as "like" or "dislike"
    like_objects = {obj_id: obj for obj_id, obj in data.items() if obj.get("label") == "like"}
    dislike_objects = {obj_id: obj for obj_id, obj in data.items() if obj.get("label") == "dislike"}

    # Randomly pick 100 objects with "label" as "like" and "dislike"
    selected_like_objects = dict(random.sample(like_objects.items(), min(num_objects, len(like_objects))))
    selected_dislike_objects = dict(random.sample(dislike_objects.items(), min(num_objects, len(dislike_objects))))

    # Merge the selected objects
    merged_objects = {**selected_like_objects, **selected_dislike_objects}

    # Write the merged objects to the output JSON file
    with open(output_file, 'w') as output:
        json.dump(merged_objects, output, indent=2)

# Replace 'your_file.json' and 'selected_objects.json' with your file names
pick_and_merge_objects('output.json', 'selected_objects.json', num_objects=100)
