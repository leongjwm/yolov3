import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import os
import tempfile
import glob
import shutil
import pandas as pd
import json

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


@torch.no_grad()
def detect(opt):
    source, weights, view_img, save_txt, imgsz, json_source = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.json
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        
    # Create temporary directories to store output

    tempSave = tempfile.TemporaryDirectory(dir=os.getcwd())
    tempSavePath = tempSave.name
    
    tempTxt = tempfile.TemporaryDirectory(dir=os.getcwd())
    tempTxtPath = tempTxt.name

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                
            p = Path(p)  # to Path
            save_path = tempSavePath + "/" + str(p.name)
            txt_path = tempTxtPath + "/" + str(p.stem)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or opt.save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                        if opt.save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
    
    
    # Create output video / image
    isImage = True
    imgCounter = max(len(glob.glob(tempSavePath + '/*.jpg')), len(glob.glob(tempSavePath + '/*.jpeg')), len(glob.glob(tempSavePath + '/*.png')))
    if imgCounter > 1: # if video
      # For combining frames into a video
      img_array = []
      isImage = False
    
      for frame in sorted(glob.glob(tempSavePath + '/*.jpg')): # select all .jpg files
        img = cv2.imread(frame)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    
      fourcc = cv2.VideoWriter_fourcc(*'DIVX')
      out = cv2.VideoWriter('OutputVideo.avi', fourcc, 15, size)
    
      for i in range(len(img_array)):
        out.write(img_array[i])
    
      cv2.destroyAllWindows()
      out.release()
    else: # for sole image
        for frame in os.listdir(tempSavePath):
            img_name = frame
            source = os.path.join(tempSavePath, frame)
            destination = os.path.join(os.getcwd(), frame)
            # move only files
            if os.path.isfile(source):
                shutil.move(source, destination)
            
        cv2.destroyAllWindows()

    
    
    # Obtain Output CSV Folder
    if json_source: # for video
        master_df = pd.DataFrame(columns=["frame_index", "no_of_ships", "no_of_kayaks", "ships_coordinates", "kayaks_coordinates"])
        with open(json_source) as f:
            jsonData = json.load(f)
            frames_to_csv = jsonData['frames_to_infer']
        frames_to_csv = list(set(frames_to_csv)) # remove duplicates
        frames_to_csv.sort()
        for frame_to_infer in frames_to_csv:
            new_image_name = str(frame_to_infer).rjust(6, '0')
            text_list = glob.glob(tempTxtPath + '/' + new_image_name + '.txt')
            if text_list:
                text_file = text_list[0]
                num_vessels = 0
                num_kayaks = 0
                vessels_xywh_conf = ""
                kayaks_xywh_conf = ""
                with open(text_file) as f:
                    for line in f:
                        predClass, x, y, w, h, confScore = line.split(" ")
                        xywh_conf = x + "_" + y + "_" + w + "_" + h + "_" + confScore.rstrip() + ";"
                        if int(predClass) == 0:
                            num_vessels += 1
                            vessels_xywh_conf += xywh_conf
                        else:
                            num_kayaks += 1
                            kayaks_xywh_conf += xywh_conf
                
                if not vessels_xywh_conf:
                    vessels_xywh_conf = '-'
                if not kayaks_xywh_conf:
                    kayaks_xywh_conf = '-'

                master_df.loc[len(master_df)] = [frame_to_infer, num_vessels, num_kayaks, vessels_xywh_conf, kayaks_xywh_conf]
            else:
                master_df.loc[len(master_df)] = [frame_to_infer, 0, 0, '-', '-']

        master_df.to_csv("OutputCSV.csv", index=False)
    
    if isImage: # for image
        master_df = pd.DataFrame(columns=["image_name", "no_of_ships", "no_of_kayaks", "ships_coordinates", "kayaks_coordinates"])
        text_list = glob.glob(tempTxtPath + '/*.txt')
        if text_list:
            text_file = glob.glob(tempTxtPath + '/*.txt')[0]
            num_vessels = 0
            num_kayaks = 0
            vessels_xywh_conf = ""
            kayaks_xywh_conf = ""
            with open(text_file) as f:
                for line in f:
                    predClass, x, y, w, h, confScore = line.split(" ")
                    xywh_conf = x + "_" + y + "_" + w + "_" + h + "_" + confScore.rstrip() + ";"
                    if int(predClass) == 0:
                        num_vessels += 1
                        vessels_xywh_conf += xywh_conf
                    else:
                        num_kayaks += 1
                        kayaks_xywh_conf += xywh_conf
            
            if not vessels_xywh_conf:
                vessels_xywh_conf = '-'
            if not kayaks_xywh_conf:
                kayaks_xywh_conf = '-'

            master_df.loc[len(master_df)] = [img_name, num_vessels, num_kayaks, vessels_xywh_conf, kayaks_xywh_conf]

        else:
            master_df.loc[len(master_df)] = [img_name, 0, 0, '-', '-']

        master_df.to_csv("OutputCSV.csv", index=False)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov3.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--json', type=str, default='', help='use json file to extract CSV file')
    opt = parser.parse_args()
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    if opt.update:  # update all models (to fix SourceChangeWarning)
        for opt.weights in ['yolov3.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt']:
            detect(opt=opt)
            strip_optimizer(opt.weights)
    else:
        detect(opt=opt)
        
