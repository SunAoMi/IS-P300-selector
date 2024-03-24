from queue import Queue
import threading
import time
import torch
import cv2
import yaml
import sys
from PIL import Image
from torchvision import transforms
import numpy as np
import shutil
from utils.datasets import letterbox
from utils.general import non_max_suppression_mask_conf
import argparse
import math
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import os
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image
from datetime import datetime
import random
from PIL import Image
from socket import *
import struct

ScreenFps = 5

ScreenLigth = 0.5

save_flag = 1

""" aomi  cuipengyu  mashuai  renninghui  qiruikang  weishuang  chenrui  liyankang"""
Subject_name = "texttupian"

"""kele shuibei pingguo xiangjiao1 xiangjiao2 shu"""
Lable_name = "xiangjiao2"


Video_path = r'cc.mp4'                  # 视频显示路径
Save_folder = r'path' + Subject_name
if save_flag == 1:
    timeget = datetime.now().strftime("_%Y_%m_%d_%H_%M")
    if not os.path.isdir(Save_folder):
        os.makedirs(Save_folder)
    Save_Sample_Path = Save_folder + r"\\Sample_" + Lable_name + timeget
    if not os.path.isdir(Save_Sample_Path):
        os.makedirs(Save_Sample_Path)
    Save_Lable_Path = Save_folder + r"\\Lablee_" + Lable_name + timeget
    if not os.path.isdir(Save_Lable_Path):
        os.makedirs(Save_Lable_Path)


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
def compute_color_for_labels(label):
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor

def check_img_size(img_size, s=32):
    new_size = make_divisible(img_size, int(s))
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size

def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

parser = argparse.ArgumentParser()
parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
# file/folder, 0 for webcam
parser.add_argument('--source', type=str, default="", help='source')
parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
# class 0 is person, 1 is bycicle, 2 is car... 79 is oven
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--evaluate', action='store_true', help='augmented inference')
parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
opt = parser.parse_args()
opt.img_size = check_img_size(opt.img_size)

out = opt.output
evaluate = opt.save_vid
if not evaluate:
    if os.path.exists(out):
        pass
        shutil.rmtree(out)
    os.makedirs(out)

cfg = get_config()
cfg.merge_from_file(opt.config_deepsort)
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)


def calculate_IoU(predicted_bound, ground_truth_bound):
    pxmin, pymin, pxmax, pymax = predicted_bound
    gxmin, gymin, gxmax, gymax = ground_truth_bound
    parea = (pxmax - pxmin) * (pymax - pymin)
    garea = (gxmax - gxmin) * (gymax - gymin)
    xmin = max(pxmin, gxmin)
    ymin = max(pymin, gymin)
    xmax = min(pxmax, gxmax)
    ymax = min(pymax, gymax)
    w = xmax - xmin
    h = ymax - ymin
    if w <= 0 or h <= 0:
        return 0

    area = w * h
    IoU = area / (parea + garea - area)

    return IoU


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open('data/hyp.scratch.mask.yaml') as f:
    hyp = yaml.load(f, Loader=yaml.FullLoader)
weigths = torch.load('yolov7-mask.pt')
model = weigths['model']
model = model.half().to(device)             # 模型半精度
_ = model.eval()

names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

q = Queue()
q2 = Queue()
Video_Send = Queue()
Video_full = Queue()
Mask_Send = Queue()

staytime = 1 / ScreenFps

screenflash = 100
sereenstaytime = 1 / screenflash

overflag = 0


def BrainDataSive():
    global overflag
    IP = "169.254.166.147"
    SERVER_PORT = 4452
    BUFFER = 1024 * 100000
    Use_Channel = 10

    dataSocket = socket(AF_INET, SOCK_STREAM)
    dataSocket.connect((IP, SERVER_PORT))

    def StartRecord():
        dataSocket.send(b'CTRL\x00\x02\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')

    def EndRecord():
        dataSocket.send(b'CTRL\x00\x02\x00\x09\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')

    StartRecord()
    RecordTime = 0
    start_sockt_flag = 0

    show_flag = 0
    while overflag == 0:
        RecodeList = []

        recved = dataSocket.recv(BUFFER)
        recvedlen = len(recved)
        if recvedlen != 20:
            continue

        recved = dataSocket.recv(BUFFER)
        formatted_time = datetime.now().strftime("_%Y_%m_%d_%H_%M_%S.%f")[:-3]
        recvedlen = len(recved)
        for i in range(int(recvedlen / 4)):
            ba = bytearray()
            ba.append(recved[i * 4 + 3])
            ba.append(recved[i * 4 + 2])
            ba.append(recved[i * 4 + 1])
            ba.append(recved[i * 4 + 0])
            RecodeList.append(struct.unpack("!f", ba)[0])
        RecordTime += 1
        show_flag += 1
        SavePath = Save_Sample_Path + r'\\' + str(RecordTime) + formatted_time + ".pt"
        SaveTensor = torch.Tensor(RecodeList)

        if save_flag == 1:
            if SaveTensor.shape[0] < 1000:
                overflag = 1
            torch.save(SaveTensor, SavePath)
        if show_flag >= 20:
            print(SavePath, SaveTensor.shape)

            show_flag = 0
    EndRecord()


SaveCount = 0

def PictureAdd(Or_p, p_p, id_get):
    global SaveCount
    if save_flag == 1:
        SaveCount += 1
        Or_p = cv2.resize(Or_p, (495, 270))
        p_p = cv2.resize(p_p, (495, 270))
        save_p = cv2.addWeighted(Or_p, 1, p_p, 0.5, 0)
        formatted_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S.%f_")[:-3]
        Lable_SaveName = str(formatted_time) + str(SaveCount) + "_" + str(id_get)
        SavePath = Save_Lable_Path + r'\\' + Lable_SaveName + '.jpg'
        cv2.imwrite(SavePath, save_p)

        print( SavePath)

def showdef():
    global overflag
    showflaag = 0
    starttime = time.time()

    # out_win = "output_style_full_screen"
    # cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty(out_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    screen_res = 1600, 900

    # pygame.init()
    # screen = pygame.display.set_mode((600, 600))
    # pygame.display.set_caption('screenshow')
    show_id = 0
    Show_id_time = 0.1
    split_id_time = 0.3
    show_id_dict = dict()
    flash_time_skip = 1.5
    Show_print_flag = 0
    while overflag == 0:
        if showflaag == 0:
            time.sleep(0.1)

        # 原始图像读取
        try:
            image = Video_full.get(timeout=0)
            # time.sleep(0.01)
            Video_full.queue.clear()
            VideoHigh, ViedoLow, fs = image.shape
            # print('C1')
        except:
            # print('E1')
            pass
        try:
            data = Mask_Send.get(timeout=0)
            id_list = data[0]
            pintshape = data[1]
            showflaag = 1
        except:
            pass
        if showflaag == 1:
            try:
                pnimg = cv2.resize(image.copy(), (pintshape[1], pintshape[0])) * 0
                if show_id == 0:
                    id_time_record = [0]
                    keyNumList = [0]
                    for key in id_list:

                        if key not in show_id_dict:
                            # 目标显示为白色且可以显示
                            if id_list[key][0] == 0 and len(id_list[key]) == 2:
                                color = (255, 255, 255)
                                pnimg[id_list[key][1]] = np.array(color, dtype=np.uint8)
                                PictureAdd(image.copy(), pnimg.copy(), key)
                                show_id_dict[key] = time.time()
                                show_id = key
                                break
                        else:
                            if id_list[key][0] == 0 and len(id_list[key]) == 2:
                                keyNumList.append(key)
                                id_time_record.append(time.time() - show_id_dict[key])
                    if show_id == 0 and len(id_list) != 0:
                        maxtime = max(id_time_record)
                        maxkey = keyNumList[id_time_record.index(maxtime)]

                        if maxkey != 0 and maxtime != 0:
                            if maxtime > flash_time_skip:
                                if id_list[maxkey][0] == 0 and len(id_list[maxkey]) == 2:
                                    color = (255, 255, 255)
                                    # color = (0, 0, 255)
                                    # time.sleep(0.5)
                                    pnimg[id_list[maxkey][1]] = np.array(color, dtype=np.uint8)
                                    show_id = maxkey
                                    show_id_dict[maxkey] = time.time()
                                    PictureAdd(image.copy(), pnimg.copy(), maxkey)
                            elif maxtime < 1:
                                pass
                            else:
                                keyindex = random.randint(1, len(id_time_record))
                                flashkey = keyNumList[keyindex]
                                if id_list[flashkey][0] == 0 and len(id_list[flashkey]) == 2:
                                    color = (255, 255, 255)
                                    # color = (0, 255, 0)
                                    # time.sleep(0.2)
                                    pnimg[id_list[flashkey][1]] = np.array(color, dtype=np.uint8)
                                    show_id = flashkey
                                    show_id_dict[flashkey] = time.time()
                                    PictureAdd(image.copy(), pnimg.copy(), flashkey)

                else:
                    if time.time() - show_id_dict[show_id] < Show_id_time:
                        if id_list[show_id][0] == 0:
                            if len(id_list[show_id]) == 2:
                                color = (255, 255, 255)
                                # color = (0, 255, 0)
                                pnimg[id_list[show_id][1]] = np.array(color, dtype=np.uint8)
                                split_id_time = random.uniform(2, 4) / 10
                    elif time.time() - show_id_dict[show_id] < split_id_time:
                        pass
                    else:
                        show_id = 0

                pnimg = cv2.resize(pnimg, (ViedoLow, VideoHigh))
                image = cv2.add(image, pnimg)
                scale_width = screen_res[0] / image.shape[1]
                scale_height = screen_res[1] / image.shape[0]
                scale = min(scale_width, scale_height)
                window_width = int(image.shape[1] * scale)
                window_height = int(image.shape[0] * scale)
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('image', window_width, window_height)
                cv2.imshow('image', image)
                cv2.waitKey(1)
                Show_print_flag += 1
                if Show_print_flag >= 100:
                    print(round(1 / (time.time() - starttime), 2), len(id_list))
                    Show_print_flag = 0
                starttime = time.time()
            except Exception as e:
                print("showdef()_E3", e)
                pass
    print("showdef()")
    cv2.destroyAllWindows()

def flashdef():
    ID_record_dict = dict()
    global overflag
    inputtime = time.time()
    fpsTime = time.time()
    calfps = 0
    image_counr = 0
    offset = (0, 0)
    id_list = dict()
    videogettime = time.time()
    id_now = []
    histor_id = []
    VideoMaskHigh, VideoMaskLow, fs2 = 100, 100, 3
    VideoHigh, ViedoLow, fs = 100, 100, 3
    showflag = 0
    while overflag == 0:
        # if showflag == 0:
        #     time.sleep(0.2)
        id_this = []

        try:
            data = q2.get(timeout=0)
            q2.queue.clear()
            image_get = data[0]
            # VideoMaskHigh, VideoMaskLow, fs2 = image_get.shape
            calfps = data[1]
            pred = data[2]
            pred_masks = data[3]
            if pred != None:

                nb, _, height, width = image_get.shape
                bboxes = Boxes(pred[:, :4])
                original_pred_masks = pred_masks.view(-1, hyp['mask_resolution'], hyp['mask_resolution'])
                pred_masks = retry_if_cuda_oom(paste_masks_in_image)(original_pred_masks, bboxes, (height, width),
                                                                     threshold=0.5)
                pred_masks_np = pred_masks.detach().cpu().numpy()
                pred_cls = pred[:, 5].detach().cpu().numpy()
                pred_conf = pred[:, 4].detach().cpu().numpy()
                nimg = image_get[0].permute(1, 2, 0) * 255
                nimg = nimg.cpu().numpy().astype(np.uint8)
                nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
                nbboxes = bboxes.tensor.detach().cpu().numpy().astype(int)
                pnimg = nimg.copy()

                # pnimg = pnimg * 0.0
                VideoMaskHigh, VideoMaskLow, fs2 = pnimg.shape
                pnimg_new = np.zeros((VideoMaskHigh, VideoMaskLow, fs2))
                # print('type(pnimg)', type(pnimg))

                showflag = 1

                xywh_bboxs = []
                confs = []
                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in pred:
                    # to deep sort format
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])
                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)
                outputs = deepsort.update(xywhs, confss, pnimg)
                pnimg = pnimg * 0
                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    # 物体追踪方框
                    count0 = 0
                    count1 = 0
                    cclist = []
                    ddlist = []

                    for i, box in enumerate(bbox_xyxy):
                        # cclist.append(box.tolist())
                        count0 += 1
                        x1, y1, x2, y2 = [int(i) for i in box]
                        x1 += offset[0]
                        x2 += offset[0]
                        y1 += offset[1]
                        y2 += offset[1]
                        # box text and bar
                        id = int(identities[i]) if identities is not None else 0
                        color = compute_color_for_labels(id)
                        label = '{}{:d}'.format("", id)


                        if id not in id_list:
                            id_list[id] = [0]
                        if id not in ID_record_dict:
                            ID_record_dict[id] = 1
                        else:
                            ID_record_dict[id] = ID_record_dict[id] + 1

                        id_this.append(id)

                        for one_mask, bbox, cls, conf in zip(pred_masks_np, nbboxes, pred_cls, pred_conf):
                            count1 += 1
                            if conf < 0.5:
                                continue
                            ddlist.append(bbox.tolist())
                            if calculate_IoU(box.tolist(), bbox.tolist()) < 0.8:
                                continue
                            if len(id_list[id]) == 1:
                                id_list[id].append(one_mask.copy())
                            else:
                                id_list[id][1] = one_mask.copy()

                            # color = compute_color_for_labels(id)
                            color = (255, 255, 255)
                            # pnimg = pnimg * 0
                            # print(color)
                            pnimg[id_list[id][1]] = pnimg[id_list[id][1]] * 0 + np.array(color, dtype=np.uint8) * 0.5
                            # cv2.imshow('YOLOv7 mask', pnimg)
                            # cv2.waitKey(1)

                    histor_id = id_this.copy()

                del_list = []

                for key in id_list:
                    if key not in id_this:
                        id_list[key][0] = id_list[key][0] + 1
                        if id_list[key][0] >= 3:
                            del_list.append(key)
                    else:
                        id_list[key][0] = 0
                    if id_list[key][0] == 0:
                        pass
                # 删除数据
                for key_name in del_list:
                    id_list.pop(key_name)
                Mask_Send.put([id_list, pnimg.shape])
                videogettime = time.time()

        except Exception as e:
            pass
            # print("flashdef()", e)
        if showflag == 1:
            time.sleep(0.01)



def producer():
    global overflag
    inputstarttime = time.time()
    image = None

    fpsTime2 = time.time()
    while overflag == 0:
        try:
            image = Video_Send.get(timeout=0)
            # time.sleep(0.01)
            Video_Send.queue.clear()
        except:
            image = None
        try:

            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))
            image = image.to(device)
            image = image.half()
            # print("E3")
            with torch.no_grad():
                output = model(image)
            inf_out, train_out, attn, mask_iou, bases, sem_output = output['test'], output['bbox_and_cls'], output[
                'attn'], \
                                                                    output['mask_iou'], output['bases'], output['sem']
            # print("E4")
            bases = torch.cat([bases, sem_output], dim=1)
            nb, _, height, width = image.shape
            names = model.names
            pooler_scale = model.pooler_scale
            pooler = ROIPooler(output_size=hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1,
                               pooler_type='ROIAlignV2', canonical_level=2)
            # print("E5")
            output, output_mask, output_mask_score, output_ac, output_ab = \
                non_max_suppression_mask_conf(inf_out, attn, bases, pooler, hyp, conf_thres=0.25, iou_thres=0.65,
                                              merge=False, mask_iou=None)
            # print(type(output2))
            pred, pred_masks = output[0], output_mask[0]
            calfps = round(1 / (time.time() - fpsTime2), 2)
            fpsTime2 = time.time()
            # q.put([image_, calfps])
            q.put([1])
            q.task_done()
            q2.put([image, calfps, pred, pred_masks])
            q2.task_done()
        except Exception as e:
            print("producer()", e)
            q.put([1])
            q.task_done()
            q2.put([1, None, None, None])
            q2.task_done()


def consumer():
    global overflag

    inputtime = time.time()

    cap = cv2.VideoCapture(Video_path)
    # cap = cv2.VideoCapture(0)
    if (cap.isOpened() == False):
        exit(-1)


    frame_width = 1920 // 2
    frame_height = 1080 // 2

    vid_write_image = letterbox(cap.read()[1], (frame_width, frame_height), stride=64, auto=True)[0]
    resize_height, resize_width = vid_write_image.shape[:2]
    starttime = 0

    fpscount = 0
    fpsNow = 0
    Viedo_Send_Flag = 1

    fpsTime2 = time.time()
    ccflash = 0
    fpsTime = time.time()
    histor_flag, histor_image = cap.read()
    while cap.isOpened and histor_flag and overflag == 0:
        # print("histor_flag", histor_flag)
        flag, image = histor_flag, histor_image
        if time.time() - fpsTime < staytime:
            pass
        else:
            fpsTime = time.time()
            histor_flag, histor_image = cap.read()
            image_ = image.copy()
            Video_full.put(image_)

        if flag:
            #
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = letterbox(image, frame_width, stride=64, auto=True)[0]
            try:
                if q.qsize() > 0:

                    if Viedo_Send_Flag == 1:
                        Video_Send.put(image)
            except Exception as e:
                ccflash = 0
                print("consumer()", e)

    overflag = 1
    time.sleep(2)
    sys.exit()


if __name__ == '__main__':
    Brain_flag = 1
    t = threading.Thread(target=producer)
    t1 = threading.Thread(target=consumer)
    t2 = threading.Thread(target=flashdef)
    t3 = threading.Thread(target=showdef)
    if Brain_flag == 1:
        t4 = threading.Thread(target=BrainDataSive)


    t.start()
    t1.start()
    t2.start()
    t3.start()
    if Brain_flag == 1:
        t4.start()

    t.join()
    t1.join()
    t2.join()
    t3.join()
    if Brain_flag == 1:
        t4.join()
