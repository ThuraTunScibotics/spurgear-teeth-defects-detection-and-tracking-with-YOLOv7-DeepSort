import argparse
import logging
import time
from pathlib import Path

import cv2
import imutils
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.save_csv import save_result
from utils.frame_text import put_counted_result
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# For Sort tracking
import skimage
from sort import *
from collections import deque

# Data structure as Global Variable #
# id lists to store tracked id of each detected label
array_ids_good = []
array_ids_defect = []
array_ids_total = []

#.... Initialize SORT .... 
#......................... 
sort_max_age = 5 
sort_min_hits = 2
sort_iou_thresh = 0.2
sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh)

total_counting, defect_counting, good_counting = 0, 0, 0
modulo_counting_t, modulo_counting_d, modulo_counting_g = 0, 0, 0



#....... For Counter which will count the gear .......#
def storing_tracked_id(img, bbox, identities=None, categories=None, names=None, conf_score=None, offset=(0, 0)):
    """
    This function store tracked id of detected label into respective id lists when mid point of detected bounding box is within ROI (between entry and exit coordinate)
    """
    for i, box in enumerate(bbox):
        print(i)
        print(box)
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        xyxy = [x1, y1, x2, y2]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        
        label_cat = names[cat]

        colors = [[115, 255, 50], [0, 0, 255]]
        label = f'#{id} ' + label_cat + " " + conf_score
        label_name = f'{names[cat]}'
        plot_one_box(xyxy, img, label=label, color=colors[cat], line_thickness=5)
        

        # c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        # Midpoint Of Bbox
        midpoint_x = x1+((x2-x1)/2)
        midpoint_y = y1+((y2-y1)/2)
        center_point = (int(midpoint_x),int(midpoint_y))
        midpoint_color = (255,0,0)

        # Entry Coordinate (68% of widt, 5)
        coordinateA = (int(img.shape[1] * 0.68), 5)
        # coordinateB = (int(im0.shape[1] * 0.85), int(img.shape[0]-5))
        # coordinateC = (int(im0.shape[1] * 0.95), 5)

        # Exit Coordinate (95% of width, height-5)
        coordinateD = (int(img.shape[1] * 0.95), int(img.shape[0]-5))

        if (midpoint_x > coordinateA[0] and midpoint_x < coordinateD[0]) and (midpoint_y > coordinateA[1] and midpoint_y < coordinateD[1]):
            
            midpoint_color = (0,0,255)
            print('Kategori : '+str(cat))
            
            # add total counting
            if len(array_ids_total) > 0:
                if id not in array_ids_total:
                    array_ids_total.append(id)
            else:
                array_ids_total.append(id)

            if label_name == 'Good Teeth':
                # add good_teeth counting
                if len(array_ids_good) > 0:
                    if id not in array_ids_good:
                        array_ids_good.append(id)
                else:
                    array_ids_good.append(id)

            if label_name == 'Defect Teeth':
                # add defect_teeth counting
                if len(array_ids_defect) > 0:
                    if id not in array_ids_defect:
                        array_ids_defect.append(id)
                else:
                    array_ids_defect.append(id)

        cv2.circle(img, center_point, radius=10, color=midpoint_color, thickness=5)

    return img


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace, save_csv = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.save_csv
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    if(device.type != 'cpu'):
        compute_capability = torch.cuda.get_device_capability(device=device)    
        half = (device.type != 'cpu') and (compute_capability[0] >= 8)  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

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

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    print("Names: ", names)
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    #colors = [[115, 255, 50], [0, 0, 255]]
    #print("Colors: ", colors)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    
    count_total, count_defect, count_good = 0, 0, 0

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                

                ###..................USE TRACK FUNCTION....................###
                #pass an empty array to sort
                dets_to_sort = np.empty((0,6))

                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                        np.array([x1, y1, x2, y2, conf, detclass])))
                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks =sort_tracker.getTrackers()
                
                print('Tracked Detections : '+str(len(tracked_dets)))

                #loop over tracks
                for track in tracks:
                    # color = compute_color_for_labels(id)
                    #draw tracks
                    [cv2.line(im0, (int(track.centroidarr[i][0]),
                                    int(track.centroidarr[i][1])), 
                                    (int(track.centroidarr[i+1][0]),
                                    int(track.centroidarr[i+1][1])),
                                    (225,198,0), thickness=7) 
                                    for i,_ in  enumerate(track.centroidarr) 
                                        if i < len(track.centroidarr)-1 ] 
                
                                    
                # draw boxes for visualization
                if len(tracked_dets)>0:
                    bbox_xyxy = tracked_dets[:,:4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    conf_score = f'{conf:.2f}'
                    # draw box and count
                    storing_tracked_id(im0, bbox_xyxy, identities, categories, names, conf_score)
                    print('Bbox xy count : '+str(len(bbox_xyxy)))

            # Drawing Entry Line
            coordinateA = (int(im0.shape[1] * 0.68), 5)
            coordinateB = (int(im0.shape[1] * 0.68), int(im0.shape[0])-5)
            #coordinateB = (int(im0.shape[1] * 0.85), 200)
            # Drawint Exit Line
            coordinateC = (int(im0.shape[1] * 0.95), 5)
            coordinateD = (int(im0.shape[1] * 0.95), int(im0.shape[0])-5)
            cv2.line(im0, coordinateA, coordinateB, (0, 128, 255), 6)
            cv2.line(im0, coordinateC, coordinateD, (0, 128, 255), 6)

            ### .... Get Counts by measuring the length of each track ID data structure (lists) .... ###

            # For counting TOTAL
            if (count_total == 0):
                total_counting = len(array_ids_total)
            else:
                if (total_counting < 100):
                    total_counting = len(array_ids_total)
                else:
                    total_counting = modulo_counting_t + len(array_ids_total)

                    # if the array_ids_total's length is 100, clear the array
                    if (len(array_ids_total)%100 == 0):
                        modulo_counting_t = modulo_counting_t + 100  # update modulo_counting_t with 100, becuase list length is 100
                        array_ids_total.clear()

            # For counting DEFECTS
            if (count_defect == 0):
                defect_counting = len(array_ids_defect)
            else:
                if (defect_counting < 100):
                    defect_counting = len(array_ids_defect)
                else:
                    defect_counting = modulo_counting_d + len(array_ids_defect)
                    if (len(array_ids_defect)%100 == 0):
                        modulo_counting_d = modulo_counting_d + 100
                        array_ids_defect.clear()

            # For counting GOODS
            if (count_good == 0):
                good_counting = len(array_ids_good)
            else:
                if (good_counting < 100):
                    good_counting = len(array_ids_good)
                else:
                    good_counting = modulo_counting_g + len(array_ids_good)
                    if (len(array_ids_good)%100 == 0):
                        modulo_counting_g = modulo_counting_g + 100
                        array_ids_good.clear()
            
            # Put the detected & counted information on the frame
            text1 = 'Defect Products = '+str(defect_counting)
            text2 = 'Good Products = '+str(good_counting)
            text3 = 'Total Products = '+str(total_counting)
            put_counted_result(im0, text1, text2, text3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            
            # Stream results
            if view_img:
                # Set the video size
                im_v = cv2.resize(im0, (640, 440), interpolation = cv2.INTER_CUBIC)
                cv2.imshow(str(p), im_v)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
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
                            fps, w, h = 20, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
        
    print("\n no. of defects: ", defect_counting)
    print("\n no. of good: ", good_counting)
    print("\n no. of total: ", total_counting)
    
    # Saving detected results in CSV file
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

    if save_csv:
        save_result(defect_counting, good_counting, total_counting, weekdays)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-csv', action='store_true', help='save predicted results to *.csv')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
