from pathlib import Path
import sys
import os
import queue
from res.video_stream import VideoStream
# from torch.utils.data import Dataset, DataLoader

import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
print(ROOT)

class Camera():
    """
    Simulate the video streams, load local video files and send data stream to the edge side.
    """
    def __init__(self):
        self.video_path = os.path.join(ROOT, 'res', 'video_noblack_easy_mini')
        self.video_pathlist = os.listdir(self.video_path)
        self.video_pathlist.sort(key=lambda fn: list(map(ord, fn)))
        # print(self.video_pathlist)
        file = open(os.path.join(ROOT, 'res', 'all_video_names_easy.txt'), 'w', encoding='utf-8')
        file.write(str(self.video_pathlist))
        file.close()

        self.chunk_duration = 1

        self.cache = queue.Queue()

    def __len__(self):
        return len(self.video_pathlist)

    def __getitem__(self, idx):
        video_name = self.video_pathlist[idx]
        video_path = os.path.join(ROOT, 'res', 'video_noblack_easy_mini', video_name)
        stream = VideoStream(video_path, self.chunk_duration)

        video_label_names = os.listdir(os.path.join(ROOT, 'res', 'labels_noblack_easy_mini', video_name.split('.')[0]))
        video_label_names.sort(key=lambda x: int(x[-9:-4]))

        # video_label_list = []
        # for name in video_label_names:
        #     video_label_list.append(np.loadtxt(os.path.join(ROOT, 'res', 'labels', video_name.split('.')[0], name)))
        sample = {'stream': stream, 'label_names': video_label_names}
        return sample


# camera_dataset = Camera()
# for i in range(len(camera_dataset)):
#     sample = camera_dataset[i]
#
#     break