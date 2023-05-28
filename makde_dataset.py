import os
import json
import shutil
from collections import deque
import random

def extend_frame_count(subfolder, min_video_length):
    """If a subfolder has fewer than min_video_length frames, append frames from the beginning."""
    frames = deque(sorted(os.listdir(subfolder)))
    extended_frames = list(frames)  # copy original frame order

    if len(frames)< min_video_length:
        while len(extended_frames) < min_video_length:
            extended_frames.extend(frames)

        return extended_frames
    else:
        return frames

def combine_frames(source_folder, target_folder, min_video_count, max_video_count,minimum_video_length = 10):
    """Combine frames from subfolders in source_folder into target_folder, ensuring each subfolder
    has at least min_video_length frames."""
    video_folders = deque(os.listdir(source_folder))
    total_length = len(video_folders)

    print('video_folders length', len(video_folders))
    json_data = {}
    video_start_points = []
    frame_start_point = 0
    longvideo_count = 0
    
    while video_folders:
        longvideo_count += 1
        random_video_count = random.randint(min_video_count, max_video_count)  # Generate a random number of videos
        os.makedirs(os.path.join(target_folder, str(longvideo_count)), exist_ok=True)
        json_data = {}
        video_start_points = []
        frame_start_point = 0

        while video_folders and random_video_count > 0:
            subfolder = video_folders.popleft()
            subfolder_path = os.path.join(source_folder, subfolder)

            if os.path.isdir(subfolder_path):
                extended_frames = extend_frame_count(subfolder_path, minimum_video_length)
                video_length = len(extended_frames)
                video_start_points.append(frame_start_point)  # Store start point of each new video

                for index, frame in enumerate(extended_frames):
                    # Rename the frame to be sequential in the new folder
                    new_frame_name = f"frame_{frame_start_point + index}.jpg"
                    shutil.copy(os.path.join(subfolder_path, frame), os.path.join(target_folder, str(longvideo_count), new_frame_name))

                json_data[subfolder] = {"start_frame": frame_start_point, "end_frame": frame_start_point + video_length - 1}
                frame_start_point += video_length
                random_video_count -= 1
        left_number = len(video_folders)
        if  int(left_number*100/total_length) % 2  == 0:
            print( 'left video percentage: ',  len(video_folders)*100/total_length )

        # Create a JSON file
        with open(os.path.join(target_folder, str(longvideo_count), 'frame_data.json'), 'w') as json_file:
            json.dump({"video_data": json_data, "video_start_points": video_start_points[1:]}, json_file)
        
# Define your source and target folder here:
source_folder = '/mnt/drive1/hsun/videoSeg/data/video_datasets/CondensedMovies/video_frames'
target_folder = '/mnt/drive1/hsun/videoSeg/data/video_datasets/CondensedMovies/new_videos'
min_video_count = 2  # The minimum number of videos to consider
max_video_count = 10  # The maximum number of videos to consider
minimum_video_length = 10
# Ensure the target folder exists
os.makedirs(target_folder, exist_ok=True)

combine_frames(source_folder, target_folder, min_video_count, max_video_count,  minimum_video_length = minimum_video_length)
