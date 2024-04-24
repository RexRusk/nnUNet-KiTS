import os
import json
import shutil

'''
D phase
'''
def save_json(obj, file, indent=4, sort_keys=True):
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


def maybe_mkdir_p(directory):
    directory = os.path.abspath(directory)
    splits = directory.split("/")[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i + 1])):
            try:
                os.mkdir(os.path.join("/", *splits[:i + 1]))
            except FileExistsError:
                # this can sometimes happen when two jobs try to create the same directory at the same time,
                # especially on network drives.
                print("WARNING: Folder %s already existed and does not need to be created" % directory)


def subdirs(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res

base = "/home/rusk/projects/nnUNet-KiTS/KiTS"
out = "/home/rusk/projects/nnUNet-KiTS/DATASET/nnUNet_raw/Dataset996_FYP-KiTS"

cases = subdirs(base, join=False)

maybe_mkdir_p(out)
maybe_mkdir_p(os.path.join(out, "imagesTr"))
maybe_mkdir_p(os.path.join(out, "imagesTs"))
maybe_mkdir_p(os.path.join(out, "labelsTr"))

for c in cases:
    case_id = int(c.split("_")[-1])
    if case_id < 40:
        shutil.copy(os.path.join(base, c, "A", "imaging.nii.gz"), os.path.join(out, "imagesTr", c + "_0000.nii.gz"))
        shutil.copy(os.path.join(base, c, "A", "segmentation.nii.gz"), os.path.join(out, "labelsTr", c + ".nii.gz"))
    else:
        shutil.copy(os.path.join(base, c, "A", "imaging.nii.gz"), os.path.join(out, "imagesTs", c + "_0000.nii.gz"))

json_dict = {}
"""
name: 数据集名字
dexcription: 对数据集的描述
modality: 模态，0表示CT数据，1表示MR数据。nnU-Net会根据不同模态进行不同的预处理（nnunet-v2版本改为channel_names）
labels: label中，不同的数值代表的类别(v1版本和v2版本的键值对刚好是反过来的)
file_ending: nnunet v2新加的
numTraining: 训练集数量
numTest: 测试集数量
training: 训练集的image 和 label 地址对
test: 只包含测试集的image. 这里跟Training不一样
"""
json_dict['name'] = "FYP-KiTS"
json_dict['description'] = "kidney and kidney tumor segmentation"
json_dict['tensorImageSize'] = "4D"
json_dict['reference'] = "Final year project KiTS data for nnunet v2"
json_dict['licence'] = ""
json_dict['release'] = "0.0"

json_dict['channel_names'] = {
    "0": "CT",
}
json_dict['labels'] = {
    "background": "0",
    "Tumor": "1",
    "Medulla": "2",
    "Cortex": "3"
}
json_dict['numTraining'] = len(cases)  # 应该是40例
json_dict['file_ending'] = ".nii.gz"
json_dict['numTest'] = 0
json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in cases]
# json_dict['test'] = []
save_json(json_dict, os.path.join(out, "dataset.json"))