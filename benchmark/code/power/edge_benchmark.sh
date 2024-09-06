#!/bin/bash
out=`python3 edge_noload_benchmark.py`

out=`python3 edge_encode_benchmark.py`
out=`python3 edge_encode_benchmark.py --bitrate=500`
out=`python3 edge_encode_benchmark.py --bitrate=1500`

out=`python3 edge_encode_benchmark.py --resolution=720x480`

out=`python3 edge_encode_benchmark.py --preset=superfast`

out=`python3 edge_inference_benchmark.py`
out=`python3 edge_inference_benchmark.py --modelid=01`
out=`python3 edge_inference_benchmark.py --modelid=03`