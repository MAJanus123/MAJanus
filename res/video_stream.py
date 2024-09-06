import ffmpeg
import numpy as np
from PIL import Image
import os
import time
import sys

# sys.path.append('/home/nx/bin-qian2-try')


# ffmpeg -i chunk10s.mp4 -vcodec copy -an output.mp4

DATA_PACKET_SIZE = 2560
DATA_HEADER_SIZE = 12
BATCH_SIZE = 4


def bytes2numpy(batch_size, in_bytes, height, width):
    """bytes data convert to numpy with specific shape"""
    if batch_size == 1:
        in_frame = (np
                    .frombuffer(in_bytes, np.uint8)
                    .reshape([height, width, 3])  # to [c, h, w]
                    )
    else:
        in_frame = (np
                    .frombuffer(in_bytes, np.uint8)
                    .reshape([-1, height, width, 3])  # to [n, c, h, w]
                    )
    return in_frame


class VideoStream:
    def __init__(self, file_path, chunk_duration):
        self.file_path = file_path
        ## Get information on video stream
        probe = ffmpeg.probe(self.file_path)
        stream_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        self.height = stream_info['height']
        self.width = stream_info['width']
        self.video_size = stream_info['bit_rate']
        self.video_duration = float(stream_info['duration'])
        self.num_frames = stream_info['nb_frames']

        ## Set the video chunk duration
        self.chunk_duration = chunk_duration

        self.inference_process = self._process_rawvideo
        self.transmission_process = self._process_h264
        self.test1_process = self._process_test

    def _process_h264(self, start_point, resolution='1280x720', bitrate='3000000'):
        """
        convert video stream to h264 format
        @param start_point: The time point at cutting begins
        @return:
            process
        """
        process = (ffmpeg
                   .input(filename=self.file_path, ss=start_point, t=self.chunk_duration)
                   .output('-', format='h264', loglevel='quiet', preset='ultrafast', s=resolution,
                           b=bitrate)  ## DO NOT USE codec = "copy", this will copy many extra video chunks
                   .run_async(pipe_stdout=True)
                   )
        return process

    def _process_rawvideo(self, start_point, resolution='480x270'):
        """
        convert video stream to bgr24 format
        @param start_point: The time point at cutting begins
        @return:
            process
        """
        process = (ffmpeg
                   .input(filename=self.file_path, ss=start_point, t=self.chunk_duration)
                   .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='quiet', s=resolution)  # to bgr format
                   .run_async(pipe_stdout=True)
                   )
        return process

    def _process_test(self):
        """
        convert video stream to h264 format
        @param start_point: The time point at cutting begins
        @return:
            process
        """
        process = (ffmpeg
                   .input(filename=self.file_path, ss=1, t=self.chunk_duration)
                   # .output('output.mp4', **{'b:v': 2000})  ## DO NOT USE codec = "copy", this will copy many extra video chunks
                   .output('output.mp4', b=10000)  ## DO NOT USE codec = "copy", this will copy many extra video chunks
                   .run_async(pipe_stdout=True)
                   )
        return process


def inference_process():
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/video/chunk10s.mp4'
    chunk_duration = 0.5

    stream = VideoStream(file_path, chunk_duration)
    chunk_i = 0
    tt0 = time.time()

    ffmpeg_process = stream.inference_process(chunk_i * chunk_duration)

    while True:
        # READ each batch
        # print(stream.height)
        # print(stream.width)
        in_bytes = ffmpeg_process.stdout.read(stream.width * stream.height * 3 * BATCH_SIZE)

        if not in_bytes:
            print(f">> chunk{chunk_i} >> [rgb24-ffmpeg] >> costs {round(time.time() - tt0, 4)}s")
            break
        else:
            in_frame = bytes2numpy(BATCH_SIZE, in_bytes, stream.height, stream.width)


def test_transmission_process():
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/video/chunk10s.mp4'
    chunk_duration = 0.5

    stream = VideoStream(file_path, chunk_duration)

    print(stream.video_duration)
    chunk_i = 0
    tt0 = time.time()

    ffmpeg_process = stream.transmission_process(chunk_i * chunk_duration)

    while True:
        data = ffmpeg_process.stdout.read(DATA_PACKET_SIZE - DATA_HEADER_SIZE)
        if not data:
            print(f">> chunk{chunk_i} >> [h264-ffmpeg] >> costs {round(time.time() - tt0, 4)}s")
            # offload_queue.put((chunk_i, "end0"))
            break


if __name__ == "__main__":
    # test_transmission_process()
    inference_process()