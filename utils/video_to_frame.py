import os
import sys
import cv2
import argparse
from tqdm import tqdm

def video_to_frames__ffjpeg(input_file, output_file, fps):
	with open(input_file, 'r') as fi:
		video_path = fi.readline().replace("\n","")
		frames_path = fi.readline().replace("\n","")

	if not os.path.exists(frames_path):
		os.mkdir(frames_path)

	options = f'-start_number 0 -vf fps={fps} -q:v 5'
	frame_path = os.path.join(frames_path, 'frame%06d.jpg')
	command = f'ffmpeg -i {video_path} {options} {frame_path}'
	os.system(command)

	with open(output_file, 'w') as fo:
		for frame_name in sorted(os.listdir(frames_path)):
			fo.writelines(os.path.join(os.path.abspath(frames_path),frame_name) + "\n")

def video_to_frames_cv2(input_video, output_dir, cam_name):
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	
	video_paths = []
	# for r, d, f in os.walk(input_dir):
	# 	for file in f:
	# 		if '.mp4' in file:
	# 			video_paths.append(os.path.join(r, file))

	# for video_path in video_paths:
	# 	print(video_path)

	# for video_path in video_paths:
	# video_dir_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_video))[0])
	# if not os.path.isdir(video_dir_path):
	# 	os.makedirs(video_dir_path)

	vid_cap = cv2.VideoCapture(input_video)
	num_frms, original_fps = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)), vid_cap.get(cv2.CAP_PROP_FPS)
	video_dir = os.path.join(output_dir, cam_name)
	dest_dir = os.path.join(video_dir, 'frames')
	if not os.path.exists(video_dir):
		os.mkdir(video_dir)
	if not os.path.exists(dest_dir):
		os.mkdir(dest_dir)
	## Number of skip frames
	time_stride = 1
	video_files = os.path.join(video_dir, cam_name + "_files.txt")
	all_imgs = os.path.join(output_dir, "all_images.txt")
	
	for frm_id in tqdm(range(0, num_frms, time_stride)):
		vid_cap.set(cv2.CAP_PROP_POS_FRAMES, frm_id)
		_, im = vid_cap.read()
		path_name = os.path.join(dest_dir, cam_name + '_' + str(frm_id) + '.jpg')
		cv2.imwrite(path_name, im)

		local_path = os.path.join("frames", cam_name + '_' + str(frm_id) + '.jpg')
		local_path2 = os.path.join(cam_name, local_path)	

		with open(video_files, "a+") as f:
			f.write(local_path + "\n")
		with open(all_imgs, "a+") as ff:
			ff.write(local_path2 + "\n")
		print("[INFO] write to ", path_name)

# if __name__ == "__main__":
	# video_to_frames("video_to_frames_input.txt", "video_to_frames_output.txt", 5)