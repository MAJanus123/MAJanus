import cv2
import os

# 图片文件夹路径
image_folder_root = '/data0/yubo.xuan/dataset/DETRAC-dataset_no_black/DETRAC-test-data/Test_mini_easy/'

# 视频文件保存路径和文件名
video_root = '/data0/yubo.xuan/dataset/DETRAC-dataset_no_black/DETRAC-test-data/Test_mini_easy_video/'
seqs = [s for s in os.listdir(image_folder_root)]
# 将图片转为视频
for seq in seqs:
    imgpath = image_folder_root+seq
    # 获取图片文件夹中的所有图片文件
    images = [img for img in os.listdir(imgpath) if img.endswith(".jpg")]
    images.sort(key=lambda x: int(x[-9:-4]))
    print(images)
    # 读取第一张图片，获取其宽度和高度
    frame = cv2.imread(os.path.join(imgpath, images[0]))
    height, width, _ = frame.shape

    # 定义视频编码器并创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码器
    video = cv2.VideoWriter(video_root+seq+'.mp4', fourcc, 30, (width, height))

    # 遍历图片列表，将每张图片写入视频
    for image in images:
        img_path = os.path.join(imgpath, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # 释放视频写入对象和关闭视频文件
    video.release()
    cv2.destroyAllWindows()

    print(f"视频已保存为：{video_root+seq+'.mp4'}")


# def compare_folders(folder1, folder2):
#     # 获取文件夹1中的所有文件名
#     files1 = os.listdir(folder1)
#
#     for file1 in files1:
#         file1_prefix = file1[0:8]
#         file1_path = os.path.join(folder1, file1_prefix+'.jpg')
#         file2_path = os.path.join(folder2, file1_prefix+'.txt')
#
#         # 检查两个文件是否存在
#         if not(os.path.exists(file1_path) and os.path.exists(file2_path)):
#             print(f"删除文件：{file1_path}")
#             os.remove(file1_path)
#
# for seq in seqs:
#     # 比对两个文件夹
#     img_path = image_folder_root + seq
#     label_path = "/data0/yubo.xuan/DETRAC-dataset_black/test_label_After_removal/" + seq
#     compare_folders(img_path, label_path)
