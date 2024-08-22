import os
import numpy as np
import argparse
import torch
import cv2
from PIL import Image
import jsonlines as jl
from tqdm import tqdm
from segment_anything_hq import sam_model_registry, SamAutomaticMaskGenerator
from categroy_rank import rank_categroy
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def load_model(model_type, sam_ckpt, device='cuda'):
    sam = sam_model_registry[model_type](checkpoint=sam_ckpt)
    sam.to(device)
    return sam

def get_generator(model,
    # points_per_side=32,
    # pred_iou_thresh=0.8,
    # stability_score_thresh=0.9,
    points_per_side=70,
    pred_iou_thresh=0.5,
    stability_score_thresh=0.8,
    crop_n_layers=2,
    crop_overlap_ratio=0.1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
):
    boxes_generator = SamAutomaticMaskGenerator(
    model=model,
    points_per_side=points_per_side,
    pred_iou_thresh=pred_iou_thresh,
    stability_score_thresh=stability_score_thresh,
    crop_n_layers=crop_n_layers,
    crop_overlap_ratio=crop_overlap_ratio,
    crop_n_points_downscale_factor=crop_n_points_downscale_factor,
    min_mask_region_area=min_mask_region_area,  # Requires open-cv to run post-processing
    )

    return boxes_generator

def resize_img(img, lbl, target_size):
    '''
    input:
        img: np(cv2 bgr2rgb)
        target_size: (h,w)
    output:
        resized_img
        h ratio
        w ratio
    '''
    ori_h, ori_w = img.shape[0], img.shape[1]
    tgt_w, tgt_h = target_size[1], target_size[0]
    scale_width = tgt_w / ori_w
    scale_height = tgt_h / ori_h

    resized_img = cv2.resize(img, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
    resized_lbl = cv2.resize(lbl, (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST)
    return resized_img, resized_lbl, scale_height, scale_width

def get_rank(dataset='gid', data_dir=None):
    '''
    generate class rank for current dataset, 
    the box with the lower rank will be ranked first 
    and will be prioritized during training.
    '''
    if dataset == 'gid':
        rank_dict = {
            "meadow": 0,
            "forest": 1,
            "built-up": 2,
            "farmland": 3,
            "water": 4,
            "unlabeled": 5,
        }
    elif dataset == 'urur':
        rank_dict = {
            "bareland": 0,
            "greenhouse": 1,
            "woodland": 2,
            "water": 3,
            "road": 4,
            "building": 5,
            "farmland": 6,
            "background": 7,
        }
    elif dataset == 'fbp':
        rank_dict,_,_ = rank_categroy(dataset='fbp', data_dir=data_dir)
    else:
        raise NotImplementedError("The dataset has not been implemented.")
    
    return rank_dict

def get_dataset(dataset='gid', data_dir=None):
    if dataset == 'gid':
        image_dir = os.path.join(data_dir, 'rgb_images')
        label_dir = os.path.join(data_dir, 'gid_labels')
    elif dataset == 'urur':
        image_dir = os.path.join(data_dir, 'image')
        label_dir = os.path.join(data_dir, 'label')
    elif dataset == 'fbp':
        image_dir = os.path.join(data_dir, 'rgb_images')
        label_dir = os.path.join(data_dir, 'fbp_labels')
    else:
        raise NotImplementedError("The dataset has not been implemented.")
    img_names = sorted(os.listdir(image_dir))
    lbl_names = sorted(os.listdir(label_dir))
    # get absoluted path directly
    assert len(img_names) == len(lbl_names), "expect same len for images and labels, but got images len:{}, labels len:{}".format(len(img_names), len(lbl_names))
    for i in range(len(img_names)):
        if dataset == "gid":
            lbl_names[i] = os.path.join(label_dir, img_names[i].split('.')[0]+'_5label.png')
        if dataset == "fbp":
            lbl_names[i] = os.path.join(label_dir, img_names[i].split('.')[0]+'_24label.png')
        elif dataset == "urur":
            lbl_names[i] = os.path.join(label_dir, img_names[i])
        img_names[i] = os.path.join(image_dir, img_names[i])

    return img_names, lbl_names, len(img_names)

def postprocess(ann, select_num, label):
    boxes = []
    category = []
    contain_classes = []
    ann_num = len(ann)
    if ann_num == 0:
        return None
    # print("label shape:{}".format(label.shape))
    sorted_anns = sorted(ann, key=(lambda x: x['predicted_iou']), reverse=True)
    for ann in sorted_anns[:min(select_num, ann_num)]:
        x, y, w, h = ann['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        # the box's size must not be zero
        if w <= 0 or h <= 0:
            continue
        label_box = label[y:y+h, x:x+w]
        # there should be at least one nonzero value in array
        non_zero_labels = label_box[label_box.nonzero()]
        if non_zero_labels.size == 0:
            continue
        most_common_value = np.argmax(np.bincount(label_box.flatten()))
        # exclude illegal elements
        if most_common_value == 255:
            continue
        total_count = np.bincount(label_box.flatten())
        # find main class
        count = total_count[most_common_value]
        percentage = (count / label_box.size)
        if percentage < 0.75:
            continue
        # find all class num
        non_zero_count = np.count_nonzero(total_count)
        # record
        contain_classes.append(non_zero_count)
        boxes.append((x,y,w,h))
        category.append(int(most_common_value))

    return boxes, category, contain_classes



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_type', type=str, default='vit_b')
    parser.add_argument('--sam_ckpt', type=str, default='../pretrained_checkpoint/sam_hq_vit_b.pth')
    parser.add_argument('--dataset', type=str, default='urur')
    parser.add_argument('--img_nums', type=int, default=5, help='the number of the image to generate boxes.')
    parser.add_argument('--start_index', type=int, default=500)
    parser.add_argument('--end_index', type=int, default=1000)
    parser.add_argument('--save_path', type=str, default='./', help='save the boxes to json')
    parser.add_argument('--data_dir', type=str, default='/data1/gyl/RS_DATASET/URUR/train')
    parser.add_argument('--select_boxes', type=int, default=500, help="select boxes for each image")
    
    args = parser.parse_args()
    # category name
    if args.dataset == 'gid':
        class_names = ['unlabeled', 'built-up', 'farmland', 'forest', 'meadow', 'water']
    elif args.dataset == 'fbp':
        class_names = [
            "unlabeled",
            "industrial area",
            "paddy field",
            "irrigated field",
            "dry cropland",
            "garden land",
            "arbor forest",
            "shrub forest",
            "park",
            "natural meadow",
            "artificial meadow",
            "river",
            "urban residential",
            "lake",
            "pond",
            "fish pond",
            "snow",
            "bareland",
            "rural residential",
            "stadium",
            "square",
            "road",
            "overpass",
            "railway station",
            "airport",
    ]
    elif args.dataset == 'urur':
        class_names = ["background", "building", "farmland", "greenhouse", "woodland", "bareland", "water", "road"]
    # define the save file
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    jsonl_file_path = os.path.join(args.save_path, f"{args.dataset}_{args.start_index}_{args.end_index}_boxes.jsonl")
    # load sam model and generator
    sam_model = load_model(args.model_type, args.sam_ckpt)
    generator = get_generator(model=sam_model)
    # get dataset
    img_names, lbl_names, img_num = get_dataset(args.dataset, args.data_dir)
    # generate boxes and postprocessing
    with torch.no_grad():
        with jl.open(jsonl_file_path, 'a') as f:
            for i in tqdm(range(args.start_index, min(img_num, args.end_index))):
                img_path = img_names[i]
                lbl_path = lbl_names[i]
                img_name = img_path.split('/')[-1]
                lbl_name = lbl_path.split('/')[-1]
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if args.dataset == 'gid' or args.dataset == 'fbp':
                    label = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
                elif args.dataset == 'urur':
                    label = Image.open(lbl_path)
                    label = np.array(label)
                image, label, scale_height, scale_width = resize_img(image, label, (2048, 2048))
                ann = generator.generate(image, multimask_output=False)
                boxes, category, contain_classes = postprocess(ann, args.select_boxes, label)
                # get final boxes coordinates
                for i, box in enumerate(boxes):
                    x,y,w,h = box
                    x = int(x / scale_width)
                    y = int(y / scale_height)
                    w = int(w / scale_width)
                    h = int(h / scale_height)
                    boxes[i] = (x,y,w,h)
                # get category name for every box
                category_names = [class_names[i] for i in category]
                # get rank for every box
                rank_dict = get_rank(args.dataset, args.data_dir)
                ranks = [rank_dict[class_name] for class_name in category_names]
                # collect boxes information to form a dict
                # Box information includes: 
                # coordinates, category id, category name, rank, and category richness
                boxes_info = []
                for i, box in enumerate(boxes):
                    boxes_info.append({
                        "coordinates": box,
                        "class_id": category[i],
                        "category": category_names[i],
                        "rank": ranks[i],
                        "richness": contain_classes[i], 
                    })
                data = {
                    "image_name": img_name,
                    "boxes": boxes_info,
                }
                f.write(data)
    

    
        








