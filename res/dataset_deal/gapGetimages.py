import os
import shutil

seqs = ['MVI_39031', 'MVI_39361', 'MVI_39371', 'MVI_39401', 'MVI_40701', 'MVI_40771', 'MVI_40775', 'MVI_40854']
images_root_path = '/data0/yubo.xuan/dataset/DETRAC-dataset_no_black/DETRAC-test-data/Insight-MVT_Annotation_Test/'
labels_root_path = '/data0/yubo.xuan/Video-Analytics-Task-Offloading_MAPPO_nolatency_separate_v8/res/labels_noblack/'
target_images_root_path = '/data0/yubo.xuan/dataset/DETRAC-dataset_no_black/DETRAC-test-data/Test_mini_easy/'
target_labels_root_path = '/data0/yubo.xuan/Video-Analytics-Task-Offloading_MAPPO_nolatency_separate_v8/res/labels_noblack_easy_mini/'

def compare_folders(folder1, folder2):
    # 获取文件夹1中的所有文件名
    files1 = os.listdir(folder1)

    for file1 in files1:
        file1_prefix = file1[0:8]
        file1_path = os.path.join(folder1, file1_prefix+'.txt')
        file2_path = os.path.join(folder2, file1_prefix+'.jpg')

        # 检查两个文件是否存在
        if not(os.path.exists(file1_path) and os.path.exists(file2_path)):
            print(f"删除文件：{file1_path}")
            os.remove(file1_path)

# for seq in seqs:
#     imgpath = images_root_path + seq
#     # 获取图片文件夹中的所有图片文件
#     images = [img for img in os.listdir(imgpath) if img.endswith(".jpg")]
#     images.sort(key=lambda x: int(x[-9:-4]))
#     target_images_path = target_images_root_path + seq
#     if not os.path.exists(target_images_path):
#         # 如果不存在，则创建文件夹
#         os.makedirs(target_images_path)
#         print(f"文件夹 '{target_images_path}' 创建成功！")
#     for index,image_name in enumerate(images):
#         if index % 3 == 0:
#             source_file_path = os.path.join(imgpath, image_name)
#             target_file_path = os.path.join(target_images_path, image_name)
#             shutil.copy2(source_file_path, target_file_path)
#     img_path = target_images_path
#     label_path = os.path.join(target_labels_root_path, seq)
#     compare_folders(label_path, img_path)


for seq in seqs:
    target_file_path = os.path.join(target_images_root_path, seq)
    images = [img for img in os.listdir(target_file_path) if img.endswith(".jpg")]
    label_path = os.path.join(target_labels_root_path, seq)
    labels = [img for img in os.listdir(label_path) if img.endswith(".txt")]
    print("images num=", len(images), "  labels num==", len(labels))