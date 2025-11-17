""" Python script to compute metrics (either CVVDP or CGVQM) """
import os
import json
from enum import Enum


## Config for CGVQM
import sys
import time
import numpy as np
import glob # Used for finding files easily


## Convig for ColorVideoVDP (CVVDP)S
import subprocess
import shlex
import re 

REF_VID = '16SSAA.mp4'
BASE_MP4 = 'data/'
BASE_FRAMES = 'data/frames/'
FRAMES_SUFFIX = '%04d.png'


class Metric(Enum):
    """Available video quality metrics."""
    CVVDP = "ColorVideoVDP"
    CGVQM = "CGVQM"

# Needed for CGVQM 
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'src', 'cgvqm'))
from cgvqm.cgvqm import run_cgvqm, visualize_emap, CGVQM_TYPE

CGVQM_CONFIG = {
    'cgvqm_type': 'CGVQM_2', # Will be converted to CGVQM_TYPE.CGVQM_2
    'device': 'cuda',        # Change to 'cpu' if no CUDA GPU is available
    'patch_scale': 4,        # Increase this value if low on available GPU memory
    'patch_pool': 'mean'     # Choose from {'max', 'mean'}
}

def get_paths(folder_name: str, metric: Metric):
    """ returns path for folder containing videos (or frames) and error map"""
    if (metric == Metric.CGVQM):
        video_path = os.path.join(project_root, BASE_MP4, folder_name)
        err_map_path = os.path.join(project_root, 'outputs/scores_cgvqm', folder_name, "scores.json")
    else:
        video_path = os.path.join(project_root, BASE_FRAMES, folder_name)
        err_map_path = os.path.join(project_root, 'outputs/scores_cvdp',folder_name, "scores.json")
    return video_path, err_map_path