import os

import cv2
FLAG = 0
if FLAG == 1:
    ##train
    seq_root = "/data0/yubo.xuan/DETRAC-dataset_black/DETRAC-train-data/Insight-MVT_Annotation_Train/"  # 图片
    xml_root = "/data0/yubo.xuan/DETRAC-dataset_black/DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML/"  # 原始xml标注
else:
    ##test
    seq_root = "/data0/yubo.xuan/DETRAC-dataset_black/test_label/"  # label
    xml_root = "/data0/yubo.xuan/DETRAC-dataset_black/DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML/"  # 原始xml标注
new_root = "/data0/yubo.xuan/DETRAC-dataset_black/test_label_After_removal/"  # label
#  yolo predict model=yolov8m.pt source=/data0/yubo.xuan/DETRAC-dataset/DETRAC-train-data/Insight-MVT_Annotation_Train/MVI_20011/ imgsz=640


# 将ignore区域涂黑
# f = open(r'/data0/yubo.xuan/DETRAC-dataset_black/test_ign.txt')
# lines = f.readlines()
# for line in lines:
#     line = line[0:-1]
#     ign_list = line.split(' ')
#     if len(ign_list) > 0:
#         name = ign_list[0]
#         path = seq_root + name + '/'
#         files_list = os.listdir(path)
#         round = int((len(ign_list) - 1) / 4)
#         for img_path in files_list:
#             img = cv2.imread(path + img_path)
#             for i in range(round):
#                 y1 = int(float(ign_list[i*4+2]))
#                 y2 = int(float(ign_list[i*4+4]))
#                 x1 = int(float(ign_list[i*4+1]))
#                 x2 = int(float(ign_list[i*4+3]))
#                 img[y1:y2, x1:x2, :] = (0, 0, 0)
#             # save figure
#             save_path = os.path.join('/data0/yubo.xuan/DETRAC-dataset/DETRAC-test-data/Insight-MVT_Annotation_Test/'+name, img_path)
#             cv2.imwrite(save_path, img)
#             print(save_path)
#

def compare(label,region):
    x1, y1, x2, y2 = label
    ignore_label = False
    ignore_x1, ignore_y1, ignore_x2, ignore_y2 = region
    if x1 >= ignore_x1 and y1 >= ignore_y1 and x2 <= ignore_x2 and y2 <= ignore_y2:
        ignore_label = True
    return ignore_label

# 去掉label在涂黑区域内的标签
with open('/data0/yubo.xuan/DETRAC-dataset_black/test_ign.txt', 'r') as f:
    lines = f.readlines()
for line in lines:
    line = line[0:-1]
    ign_list = line.split(' ')
    if len(ign_list) > 0:
        name = ign_list[0]
        labels_path = os.path.join(seq_root, name)
        new_labels_path = os.path.join(new_root, name)
        isExists = os.path.exists(new_labels_path)
        if not isExists:  # 判断如果文件不存在,则创建
            os.makedirs(new_labels_path)
        labels = [s for s in os.listdir(labels_path)]
        round = int((len(ign_list) - 1) / 4)
        for label_name in labels:
            label_path = os.path.join(seq_root, name, label_name)
            label_path_save = os.path.join(new_labels_path, label_name)
            with open(label_path_save, "w") as file:
                print(label_path_save, "空的txt文件已创建。")
            with open(label_path, 'r') as f:
                lines_label = f.readlines()
                for line_label in lines_label:
                    line_label = line_label[0:-1]
                    label_cls = line_label.split(' ')
                    cls = int(label_cls[0])
                    label = label_cls[1:]
                    label = [float(b) for b in label]
                    label_wh = [label[0] * 960, label[1] * 540, label[2] * 960, label[3] * 540]
                    label_x1 = label_wh[0] - label_wh[2] / 2
                    label_y1 = label_wh[1] - label_wh[3] / 2
                    label_x2 = label_wh[0] + label_wh[2] / 2
                    label_y2 = label_wh[1] + label_wh[3] / 2
                    for i in range(round):
                        x1 = float(ign_list[i * 4 + 1])
                        y1 = float(ign_list[i * 4 + 2])
                        x2 = float(ign_list[i * 4 + 3])
                        y2 = float(ign_list[i * 4 + 4])
                        ignore_label = compare((label_x1, label_y1, label_x2, label_y2), (x1, y1, x2, y2))
                        if not ignore_label:
                            label_str = '{:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(cls, label[0], label[1], label[2],
                                                                                    label[3])  # 宽高中心坐标归一化
                            with open(label_path_save, 'a') as f:
                                f.write(label_str)
                            break
                        else:
                            print(label_path_save)
                            break





