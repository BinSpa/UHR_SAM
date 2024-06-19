import numpy as np
from PIL import Image
import os
import cv2

def rank_categroy(dataset='fbp', data_dir=None):
    if dataset == 'fbp':
        index_dict = {
            1: 'industrial area',
            2: 'paddy field',
            3: 'irrigated field',
            4: 'dry cropland',
            5: 'garden land',
            6: 'arbor forest',
            7: 'shrub forest',
            8: 'park',
            9: 'natural meadow',
            10: 'artificial meadow',
            11: 'river',
            12: 'urban residential',
            13: 'lake',
            14: 'pond',
            15: 'fish pond',
            16: 'snow',
            17: 'bareland',
            18: 'rural residential',
            19: 'stadium',
            20: 'square',
            21: 'road',
            22: 'overpass',
            23: 'railway station',
            24: 'airport',
            0: 'unlabeled'
        }
        label_dir = os.path.join(data_dir, 'fbp_labels')
    
    label_names = os.listdir(label_dir)
    richness_dict = {}
    for i, label_name in enumerate(label_names):
        lbl_img = cv2.imread(os.path.join(label_dir, label_name), cv2.IMREAD_GRAYSCALE)
        flatten_lbl = lbl_img.flatten()
        # 计数标签中的所有类别出现的像素数量
        counts = np.bincount(flatten_lbl)
        for index, count in enumerate(counts):
            if index in richness_dict.keys():
                richness_dict[index] += count
            else:
                richness_dict[index] = count
    richness_list = []
    for key, value in richness_dict.items():
        richness_list.append((key, value))
    richness_list.sort(key=lambda x:x[1])
    # 按照instance_detect.py的设置，配置字典
    rank_dict = {}
    for rank, cate in enumerate(richness_list):
        cate_index = cate[0]
        cate_name = index_dict[cate_index]
        rank_dict[cate_name] = rank
    return rank_dict, richness_list, index_dict

if __name__ == "__main__":
    rank_dict, richness_list, index_dict = rank_categroy(data_dir="/home/rsr/gyl/RS_DATASET/FBP/train")
    for r in richness_list:
        print("{}:{}".format(index_dict[r[0]], r[1]))
    