# Detecting Vessels and Kayaks using YOLOv3

This repository is forked from Ultralytics' implementation of YOLOv3 (an object detection model) and is catered towards Vessel and Kayak Detection. This guide assumes that you have basic knowledge of Python. 

**Please read the next few sections carefully before using this repository.**

## Downloading of Weights for Vessel and Kayak Detection
**The model weights need to be downloaded from the below link and stored in the `yolov3` folder after cloning this repository.**
Due to Github's file size limit of 100MB, the weights, **`best.pt`**, need to be downloaded from **[here](https://drive.google.com/file/d/1hgV7DGNPtnOMsAjWPQ47jEooxIBjC2lg/view?usp=sharing)**.

## Setting up of Virtual Environment (Windows)
To download all the required packages to use the repository and prevent dependency conflicts, a virtual environment is required. 
This is a very basic guide on how to create virtual environments in Windows using the `venv` library will be demonstrated below. 

For Mac/Linux users, please refer to this [link](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) instead.

### Step One: Creating a Virtual Environment
Before running, we can change the directory to the path where the `yolov3` repository is located in your computer.
Run `python -m venv \path\to\myenv` using Command Prompt.

Example: `python -m venv YourVirtualEnvironment`, where `YourVirtualEnvironment` is the name of the environment. 

### Step Two: Activating your Virtual Environment 
Run `\path\to\myenv\Scripts\activate` using Command Prompt.

Example: `YourVirtualEnvironment\Scripts\activate`, where `YourVirtualEnvironment` is the name of the environment. 

### Step Three: Installing Packages in Virtual Environment 
**Remember to activate your virtual environment first**.

Run `pip install -r requirements.txt` using Command Prompt.

You now have the necessary packages needed to use the repository. 

## Detecting Vessels and Kayaks in an Image or Video
**Before using the repository, ensure that you have activated your virtual environment and also changed directories to the path where the `yolov3` repository is stored in your machine.**

**You have the choice of either processing an image or video. The image/video and additional json file (optional) must be in the `yolov3` folder.** 

File extensions supported: `*.jpg`, `*.jpeg`, `*.png` for images, `*.avi` for videos.

### Using an Image
Make a spare copy of it beforehand, as the **unprocessed image will be overwritten by the processed one in the same folder**. 

The outputs of processing an image would be: 
1. the processed image with the same filename
2. a `.csv` file with the name `OutputCSV.csv`. Both will be in the `yolov3` folder.  

### Using a Video
If using a **video**, **the original video will not be overwritten by the processed video.** **You have the option to include an additional json file stating the additional frames you would like to infer. Like an image, the information after processing these frames will be collected and stored in a .csv file.**

The outputs of processing a video would be (1) the processed video with the name `OutputVideo.avi`. If an additional json file was used, then (2) a `.csv` file with the name `OutputCSV.csv` will be produced as well. The output(s) will be located in the `yolov3` folder as well.

### Start Running

Run `python vessel_kayak_count.py` on your terminal. You wil see the following pop-up: 

Input the name (including file extension) of the image/video that you want processed, and press Enter.

Example: `YourImage.jpg` for images, `YourVideo.avi` for videos.

If your input is a video, you will have the following option of inputting a json file which specifies the frames you want to infer, 

























