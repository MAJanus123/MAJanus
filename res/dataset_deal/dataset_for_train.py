import os
import random
import shutil
import cv2

##train
seq_root = "/data0/yubo.xuan/dataset/DETRAC-dataset_no_black/DETRAC-train-data/Insight-MVT_Annotation_Train/"  # 图片
label_root = "/data0/yubo.xuan/dataset/DETRAC-dataset_no_black/train_label/"  # train label
test_root = "/data0/yubo.xuan/dataset/DETRAC-dataset_no_black/DETRAC-test-data/Insight-MVT_Annotation_Test/"
test_label_root = "/data0/yubo.xuan/dataset/DETRAC-dataset_no_black/test_label/"

#  yolo predict model=yolov8m.pt source=/data0/yubo.xuan/DETRAC-dataset/DETRAC-train-data/Insight-MVT_Annotation_Train/MVI_20011/ imgsz=640


def rename_and_move_images(source_folders, images_destination_folder, labels_destination_folder):
    if not os.path.exists(images_destination_folder):
        os.makedirs(images_destination_folder)

    if not os.path.exists(labels_destination_folder):
        os.makedirs(labels_destination_folder)
    source_folders.sort(key=lambda x: int(x[-5:-1]))
    for folder in source_folders:
        if os.path.exists(seq_root + folder):
            files = os.listdir(seq_root + folder)
            # 计算需要选择的验证集图片数量
            num_validation_images = int(len(files) * 10 / 100)
            # 随机选择验证集图片
            validation_images_filename = random.sample(files, num_validation_images)
            # 取出train图片文件名
            train_images_filename = [x for x in files if x not in validation_images_filename]
            validation_images_filename.sort(key=lambda x: int(x[-9:-4]))
            train_images_filename.sort(key=lambda x: int(x[-9:-4]))
            # train
            for file in train_images_filename:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    filename, extension = os.path.splitext(file)
                    # images
                    new_filename = folder + '_' + file
                    source_path = os.path.join(seq_root, folder, file)
                    destination_path = os.path.join(images_destination_folder,'train', new_filename)
                    shutil.copy(source_path, destination_path)

                    # labels
                    label_new_filename = folder + '_' + filename + '.txt'
                    source_path = os.path.join(label_root, folder, filename+'.txt')
                    label_destination_path = os.path.join(labels_destination_folder,'train', label_new_filename)
                    shutil.copy(source_path, label_destination_path)

            # validation
            for file in validation_images_filename:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    filename, extension = os.path.splitext(file)
                    # images
                    new_filename = folder + '_' + file
                    source_path = os.path.join(seq_root, folder, file)
                    destination_path = os.path.join(images_destination_folder,'val', new_filename)
                    shutil.copy(source_path, destination_path)

                    # labels
                    label_new_filename = folder + '_' + filename + '.txt'
                    source_path = os.path.join(label_root, folder, filename + '.txt')
                    label_destination_path = os.path.join(labels_destination_folder, 'val', label_new_filename)
                    shutil.copy(source_path, label_destination_path)
            print("successfully finsh", folder)
        else:
            print(f"Folder '{folder}' does not exist.")

    # test
    test_source_folders = [s for s in os.listdir(test_root)]
    for test_folder in test_source_folders:
        test_files = os.listdir(test_root + test_folder)
        for test_file in test_files:
            if test_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                filename, extension = os.path.splitext(test_file)
                # images
                new_filename = test_folder + '_' + test_file
                source_path = os.path.join(test_root, test_folder, test_file)
                destination_path = os.path.join(images_destination_folder, 'test', new_filename)
                shutil.copy(source_path, destination_path)

                # labels
                label_new_filename = test_folder + '_' + filename + '.txt'
                source_path = os.path.join(test_label_root, test_folder, filename + '.txt')
                label_destination_path = os.path.join(labels_destination_folder, 'test', label_new_filename)
                shutil.copy(source_path, label_destination_path)
        print("successfully finsh", test_folder)

# 指定源文件夹列表和目标文件夹路径
source_folders = [s for s in os.listdir(seq_root)]
images_destination_folder = "/data0/yubo.xuan/UA-DETRAC_NO_BLACK/images/"
labels_destination_folder = "/data0/yubo.xuan/UA-DETRAC_NO_BLACK/labels/"

# 调用函数进行重命名和移动
rename_and_move_images(source_folders, images_destination_folder, labels_destination_folder)
