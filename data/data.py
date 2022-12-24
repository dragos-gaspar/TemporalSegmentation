import logging
import os
import random
import re

import cv2
import numpy as np

from config import DataConfig as Config
from data.utils import is_overlap, draw_pair


logger = logging.getLogger('data')
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel('DEBUG')


SCENES_RE = re.compile(r'.*_scenes$')


def parse_annotation_file(path: str) -> list:
    with open(path, 'r') as f:
        contents = f.read().split()

    scenes = []

    for i in range(len(contents) // 2):
        scenes.append((int(contents[2*i]), int(contents[2*i + 1])))

    return scenes


def read_frame(cap, frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    res, frame = cap.read()

    if res:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    else:
        raise RuntimeError(f'Unable to read frame {frame_number}')


def get_frame_pair(cap, first_frame_index, second_frame_index):
    first_frame = read_frame(cap, first_frame_index)
    second_frame = read_frame(cap, second_frame_index)

    if Config.SHOW_EXTRACTED:
        draw_pair(first_frame, second_frame)

    pair = np.stack((first_frame, second_frame))

    return pair


def process_raw_dataset():

    logger.info(f'Reading files in {os.path.abspath(Config.RAW_DATA_PATH)}')

    all_file_names = sorted(os.listdir(Config.RAW_DATA_PATH))
    videos_and_annotations = []
    for file in all_file_names:
        file_name, extension = tuple(os.path.splitext(os.path.basename(file)))
        if extension == '.txt':
            if re.match(SCENES_RE, file_name):
                video_name = file_name[:-7]
                for potential_match in all_file_names:
                    if re.match(r'^' + video_name, potential_match) and potential_match != file_name + extension:
                        videos_and_annotations.append((potential_match, file_name + extension))

    logger.info(f'Checking output path {os.path.abspath(Config.PROCESSED_DATA_PATH)}')

    transitions_out_path = os.path.join(Config.PROCESSED_DATA_PATH, 'transitions')
    non_transitions_out_path = os.path.join(Config.PROCESSED_DATA_PATH, 'non-transitions')

    if not os.path.exists(transitions_out_path):
        os.makedirs(transitions_out_path)
    if not os.path.exists(non_transitions_out_path):
        os.makedirs(non_transitions_out_path)

    tr_glob_count = 0
    non_tr_glob_count = 0

    for video, annotation_file in videos_and_annotations:

        logger.info(f'Extracting frames from {video}...')

        video_capture = cv2.VideoCapture(os.path.join(Config.RAW_DATA_PATH, video))
        annotations = parse_annotation_file(os.path.join(Config.RAW_DATA_PATH, annotation_file))

        count = len(annotations) - 1
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get pairs of frames from transitions
        for i in range(count):
            pair = get_frame_pair(video_capture, annotations[i][1], annotations[i+1][0])
            np.save(os.path.join(Config.PROCESSED_DATA_PATH, f'transitions/{tr_glob_count}'), pair)
            tr_glob_count += 1

        # Get random pairs of frames from scenes
        transitions = [(annotations[i][1], annotations[i+1][0]) for i in range(count)]
        idxs = []
        while len(idxs) < count:
            idx = random.randrange(0, total_frames - 1)
            if not any([is_overlap((idx, idx + 1), tr) for tr in transitions]):
                idxs.append((idx, idx+1))

        for i, idx in enumerate(idxs):
            pair = get_frame_pair(video_capture, *idx)
            np.save(os.path.join(Config.PROCESSED_DATA_PATH, f'non-transitions/{non_tr_glob_count}'), pair)
            non_tr_glob_count += 1

    logger.info('Finished extracting frame pairs')
