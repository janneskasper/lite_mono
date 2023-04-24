import argparse
import cv2
import os
import numpy as np
import math
import argparse
import glob


STEREO_SIDE_MAP={"l": "2", "r": "3"}

def get_images_from_mp4(path, skip_frames=0) -> list:
    assert os.path.isfile(path), f"Loading video from {path} no possible because not a file!"

    vid = cv2.VideoCapture(path)

    print(f"Loading video: {path} ...")

    frames = []

    index=0
    while True:
        success, frame = vid.read()
        if not success:
            break
        if index % (skip_frames+1):
            index += 1
            continue
        frames.append(frame)
        index += 1
    return frames
    

def import_to_kitti_dataset(root_path: str, video_paths: list, dir_name: str="", file_extension="jpg", train_split=0.8, skip_frames=1, stereo_side="l"):
    assert os.path.isdir(root_path), f"Path to kitti root dir: {root_path} is not a directory!"
    assert file_extension == "jpg" or file_extension == "png", "Image file extension should be \"jpg\" or \"png\"!"

    assert not os.path.exists(os.path.join(root_path, dir_name)), f"Specified folder for saving the dataset already exists!"

    frames_per_video=[]
    for p in video_paths:
        if os.path.isfile(p):
            frames_per_video.append(get_images_from_mp4(p, skip_frames=skip_frames))
        
    train_split_filenames = []
    val_split_filenames = []

    index=0
    for video_frames in frames_per_video:
        print(f"Saving frames to: dataset_{index:06d}")
        video_dir = f"dataset_{index:06d}"
        split_file_path = os.path.join(dir_name, video_dir)
        num_train_files = math.floor(train_split * len(video_frames))

        frame_id=0
        created = False
        for frame in video_frames:
            # simulate kitti structure
            kitti_path, filename = kitti_path_format(frame_id, stereo_side=stereo_side)
            kitti_path = os.path.join(dir_name, video_dir, kitti_path)
            if not created:
                abs_path = os.path.join(root_path, kitti_path)
                if not os.path.exists(abs_path):
                    os.makedirs(abs_path)
                created = True
            # write image to folder
            cv2.imwrite(os.path.join(abs_path, filename), frame)
            
            # add image to split file, do not add frame first and last frame because training uses -1,0,1 frames 
            if frame_id > 0 and frame_id < len(video_frames)-1: 
                if frame_id < num_train_files:
                    train_split_filenames.append(kitti_split_file_format(split_file_path, frame_id, stereo_side=stereo_side))
                else:
                    val_split_filenames.append(kitti_split_file_format(split_file_path, frame_id, stereo_side=stereo_side))
            frame_id += 1
        index+=1
    
    with open(os.path.join(root_path, dir_name, "train_files.txt"), "w") as f:
        f.writelines(line + '\n' for line in train_split_filenames)
    with open(os.path.join(root_path, dir_name, "val_files.txt"), "w") as f:
        f.writelines(line + '\n' for line in val_split_filenames)

def kitti_path_format(frame_id, stereo_side="l", file_extension="jpg"):
    assert stereo_side in STEREO_SIDE_MAP.keys(), f"Kitti path format: stereo side ({stereo_side}) not valid!"
    
    return f"image_0{STEREO_SIDE_MAP[stereo_side]}/data", f"{frame_id:010d}.{file_extension}"

def kitti_split_file_format(rel_path, frame_id, stereo_side="l"):
    assert stereo_side in STEREO_SIDE_MAP.keys(), f"Kitti split file format: stereo side ({stereo_side}) not valid!"
    
    return f"{rel_path} {frame_id} {stereo_side}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Speficies the input split file path")
    parser.add_argument("-o", "--kitti_root", type=str, required=True, help="Speficies the kitti root dir path")
    parser.add_argument("-s", "--skip_frames", type=int, default=1, help="Speficies the number of frames to skip in the video")
    parser.add_argument("-f", "--video_format", type=str, default="mp4", help="Speficies the video file format")
    parser.add_argument("--sub_dir", type=str, default="custom_data", help="Speficies a sub directory to structure the files")
    args = parser.parse_args()

    data=[]

    if os.path.isfile(args.input):
        data=[args.input]
    elif os.path.isdir(args.input):
        data = glob.glob(os.path.join(args.input, '*.{}'.format(args.video_format)))
    else:
        print("Specfied input path is not valid")
        exit(0)

    if not os.path.exists(os.path.join(args.kitti_root, args.sub_dir)):
        os.makedirs(os.path.join(args.kitti_root, args.sub_dir))

    print(data)
    
    import_to_kitti_dataset(args.kitti_root, data, dir_name=args.sub_dir, skip_frames=args.skip_frames)