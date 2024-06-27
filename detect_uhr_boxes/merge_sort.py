import jsonlines as jl
import os
import argparse
from typing import List
from tqdm import tqdm

def merge(jsonl_list: List, save_dir: str):
    save_name = jsonl_list[0].split('_')[0]+'_boxes.jsonl'
    save_path = os.path.join(save_dir, save_name)
    total_dataset = []
    for jsonl_file in jsonl_list:
        jsonl_info = []
        with jl.open(os.path.join(save_dir, jsonl_file), 'r') as f:
            for line in f:
                jsonl_info.append(line)
        total_dataset = total_dataset + jsonl_info
    with jl.open(save_path, 'w') as f:
        for img_info in total_dataset:
            f.write(img_info)
    
    return save_path

def resort(jsonl_path: str):
    '''
    sorted boxes for every image
    '''
    jsonl_info = []
    with jl.open(jsonl_path, 'r') as f:
        for line in f:
            # {'image_name': string, 'boxes': list[dict]}
            jsonl_info.append(line)
    for img_info in tqdm(jsonl_info):
        img_info["boxes"] = sorted(img_info["boxes"], key=lambda x:(x["rank"], -x["richness"]))
    with jl.open(jsonl_path, 'w') as f:
        for line in jsonl_info:
            f.write(line)
    print("write sorted boxes to jsonl file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--jsonl_dir', type=str, default='jsonls')
    parser.add_argument('--dataset', type=str, default='fbp')

    args = parser.parse_args()

    if args.dataset == 'gid':
        jsonl_list = ['gid_0_150_boxes.jsonl']
    elif args.dataset == 'urur':
        jsonl_list = ['urur_0_539_boxes.jsonl', 'urur_539_1078_boxes.jsonl', 'urur_1078_1617_boxes.jsonl', 'urur_1617_3008_boxes.jsonl']
    elif args.dataset == 'fbp':
        jsonl_list = ['fbp_0_150_boxes.jsonl']
    else:
        raise NotImplementedError("The dataset has not been implemented")
    # merge all information to one jsonl file
    save_path = merge(jsonl_list, save_dir=args.jsonl_dir) 
    # sort boxes for every image
    resort(save_path)
    # check the sorted list
    with jl.open(save_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            print("{}\n".format(line["image_name"]))
            print("{}\n".format(line["boxes"]))