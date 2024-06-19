import numpy as np
from PIL import Image
import os

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
        lbl_img = Image.open(os.path.join(label_dir, label_name))
        np_lbl_img = np.array(lbl_img)
        flatten_lbl = np_lbl_img.flatten()
        counts = np.bincount(flatten_lbl)
        values = np.nonzero(counts)[0]
        counts = counts[values]
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
    rank_dict, richness_list, index_dict = rank_categroy()
    for r in richness_list:
        print("{}:{}".format(index_dict[r[0]], r[1]))
    