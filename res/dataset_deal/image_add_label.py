import os
import cv2

seq_root = "/data0/yubo.xuan/DETRAC-dataset_black/DETRAC-test-data/Insight-MVT_Annotation_Test/"  # 图片
label_root = "/data0/yubo.xuan/DETRAC-dataset_black/test_label_After_removal/"
data_have_labels = "/data0/yubo.xuan/DETRAC-dataset_black/DETRAC-test-data/test_data_noblack_have_labels2/"  # 新生成的带有标签框的图片保存目录
seqs = [s for s in os.listdir(seq_root)]
for seq in seqs:
    image_files = os.listdir(seq_root + seq)
    for image_file in image_files:
        image_path = os.path.join(seq_root, seq, image_file)
        image = cv2.imread(image_path)
        label_path = os.path.join(label_root, seq, image_file[0:-4]+'.txt')
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line[0:-1]
            label = line.split(' ')[1:]
            label = [float(b) for b in label]
            label = [label[0]*960, label[1]*540, label[2]*960, label[3]*540]
            x1 = int(label[0] - label[2]/2)
            y1 = int(label[1] - label[3]/2)
            x2 = int(label[0] + label[2]/2)
            y2 = int(label[1] + label[3]/2)
            # 添加标签框到图片上
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 这里的(0, 255, 0)是框的颜色，2是线的粗细
        isExists = os.path.exists(os.path.join(data_have_labels, seq))
        if not isExists:  # 判断如果文件不存在,则创建
            os.makedirs(os.path.join(data_have_labels, seq))
        # 保存标记后的图像
        cv2.imwrite(os.path.join(data_have_labels, seq, image_file), image)
        print(os.path.join(data_have_labels, seq, image_file))