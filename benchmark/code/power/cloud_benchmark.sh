#!/bin/bash
out=`python3 cloud_noload_benchmark.py`

out=`python3 cloud_decode_benchmark.py`

out=`python3 cloud_inference_benchmark.py`
out=`python3 cloud_inference_benchmark.py --modelid=01`
out=`python3 cloud_inference_benchmark.py --modelid=03`