import cv2
import os
import glob
import json
import tempfile
import re
import time
from shutil import copyfile

"""
For image/video file input
File extensions supported: *.jpg, *.jpeg, *.png for images, *.avi, *.mp4 for videos

Note: json file input is only required for videos.
"""
print("File extensions supported: *.jpg, *.jpeg, *.png for images, *.avi for videos")
fileInput = input("Please input the video/image file name (including file extension): ")

# Check if the string is empty:
if not fileInput:
  raise Exception('Empty string. Please input a valid video/image file name')

image_regex = "(^.+(\.(jpe?g)|(png)$))"
video_regex = "(^.+(\.(avi)|(mp4)$))"     

p1 = re.compile(image_regex)
p2 = re.compile(video_regex)

# Check if the file format is appropriate:
if not (re.search(p1, fileInput)) and not (re.search(p2, fileInput)):
  raise Exception('Please input a file name, including the appropriate file extension.')

tempFrames = tempfile.TemporaryDirectory(dir=os.getcwd())
tempPath = tempFrames.name
relativetempPath = os.path.basename(tempPath)

imageBool = re.search(p1, fileInput)
videoBool = re.search(p2, fileInput)

# Open image/video file:
if imageBool: # image file
  copyfile(os.path.join(os.getcwd(), fileInput), os.path.join(tempPath, '{}'.format(fileInput)))
else:
  frames_to_infer = []
  vidCapture = cv2.VideoCapture(fileInput)
  haveJson = input("Use json file to extract specific frames from video? (enter 'Y' or 'N') ")
  if not haveJson or (haveJson != 'Y' and haveJson != 'N'):
    raise Exception("Please input either 'Y' for Yes and 'N' for No.")
  elif haveJson == 'Y':
    jsonInput = input("Please input the json file name (including .json extension): ")
    json_regex = "(^.+(\.(json)$))"
    p3 = re.compile(json_regex)
    jsonBool = re.search(p3, jsonInput)
    if not jsonBool:
      raise Exception('Please input valid json file name.')
    with open(jsonInput) as f:
      jsonData = json.load(f)
    frames_to_infer = jsonData['frames_to_infer']
  else:
    jsonBool = False

  startTime = time.time()
  video_frames = []

  # To get output processed video from unprocessed video
  for selected_frame in range(0,int(vidCapture.get(7)),4):
    vidCapture.set(1, selected_frame) # propId = 1 is for index of frames, RHS is the frame you want to capture
    ret, frame = vidCapture.read()
    new_image_name = str(selected_frame).rjust(6, '0')
    path = os.path.join(tempPath, '{}.jpg'.format(new_image_name))
    if ret == True:
      cv2.imwrite(path, frame)
      video_frames.append(selected_frame)


  # To get output CSV from json
  if jsonBool:
    frames_to_infer = list(set(frames_to_infer)) # remove duplicates
    frames_to_infer.sort()
    for selected_frame in frames_to_infer:
      if selected_frame not in video_frames: # prevent double counting of frames
        vidCapture.set(1, selected_frame) # propId = 1 is for index of frames, RHS is the frame you want to capture
        ret, frame = vidCapture.read()
        new_image_name = str(selected_frame).rjust(6, '0')
        path = os.path.join(tempPath, '{}.jpg'.format(new_image_name))
        if ret == True:
          cv2.imwrite(path, frame)
          video_frames.append(selected_frame)

if imageBool:
  os.system('python custom_detect.py --source {} --weights best.pt --save-txt --save-conf'.format(relativetempPath))
elif videoBool and not jsonBool:
  os.system('python custom_detect.py --source {} --weights best.pt --save-txt --save-conf'.format(relativetempPath))
  endTime = time.time()
  print("Time taken in total: " + str(endTime-startTime))
else:
  os.system('python custom_detect.py --source {} --weights best.pt --save-txt --save-conf --json {}'.format(relativetempPath, jsonInput))
  endTime = time.time()
  print("Time taken in total: " + str(endTime-startTime))

