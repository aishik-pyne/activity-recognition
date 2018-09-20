import json, os
from utils import streamer, frameCount
config = json.load(open("config.json", "r"))
kth_dataset_path = config["paths"]["kth_dataset"]

# List activities
activities = os.listdir(kth_dataset_path)
files = {}
for activity in activities:
    files[activity] = list(filter(lambda x: x.endswith(".avi"), [_file for _file in os.listdir(os.path.join(kth_dataset_path, activity))]))
    
print(frameCount(os.path.join(kth_dataset_path, activities[0], files[activities[0]][0])))
