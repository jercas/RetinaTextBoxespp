import os
import cv2
import zipfile
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from augmentations import Augmentation_inference
from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw

os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1", "Yes", "Y", "True", "T")

parser = argparse.ArgumentParser(description='PyTorch RetinaNet Evaluating')
parser.add_argument('--input_size', '-i', default=1280,
                                                                    type=int, help='Model input size')
parser.add_argument('--cls_thresh', '-c', default=0.4,
                                                                    type=float, help='Classification threshold')
parser.add_argument('--nms_thresh', '-n', default=0.1,
                                                                    type=float, help='NMS threshold')
parser.add_argument('--dataset', '-d', default='ICDAR2015',
                                                                    type=str, help='evaluation dataset')
parser.add_argument('--tune_from', '-t', default='./model/ICDAR2015_TextBoxes.pth',
                                                                    type=str, help='pre-trained weight')
parser.add_argument('--output_zip', '-o', default='_result',
                                                                    type=str, help='evaluation zip output')
parser.add_argument('--save_img', '-s', default=1,
                                                                    type=str2bool, help='save output image')
args = parser.parse_args()

net = RetinaNet()
net = net.cuda()

# load checkpoint
checkpoint = torch.load(args.tune_from)

net.load_state_dict(checkpoint['net'])
net.eval()

encoder = DataEncoder(args.cls_thresh, args.nms_thresh)

# test image path & list
img_dir = "./DB/ICDAR2015/test/" if args.dataset in ["ICDAR2015"] else "./DB/ICDAR2013/test/"
val_list = [im for im in os.listdir(img_dir) if "jpg" in im]

if not os.path.exists(args.output_zip):
    os.mkdir(args.output_zip)

# save results dir & zip
eval_dir = "icdar2015" if args.dataset in ["ICDAR2015"] else "icdar2013"
result_zip = zipfile.ZipFile(eval_dir + args.output_zip, 'w')

_multi_scale = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1056, 1088, 1120, 1152, 1184, 1216, 1248, 1280]
multi_scale = np.array(_multi_scale)

for n, _img in enumerate(val_list):
    #print("infer : %d / %d" % (n, len(val_list)), end='\r')
    save_file = "res_%s.txt" % (_img[:-4])
    print("save [%d/%d] %s" % (n, len(val_list), save_file), end='\r')
    f = open(args.output_zip + "/res_%s.txt" % (_img[:-4]), "w")

    img = cv2.imread(img_dir + _img)
    height, width, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    input_scale = args.input_size

    if args.dataset in ["ICDAR2013"]:
        shorter_side = min(height, width)
        idx = multi_scale.searchsorted(shorter_side)
        idx = np.clip(idx, 0, len(_multi_scale)-1)
        input_scale = _multi_scale[idx]

    x,_,_ = Augmentation_inference(input_scale)(img)
    x = x.unsqueeze(0)
    x = Variable(x)
    x = x.cuda()

    # prediction
    loc_preds, cls_preds = net(x)

    quad_boxes, labels, scores = encoder.decode(loc_preds.data.squeeze(0), cls_preds.data.squeeze(0), input_scale)
    if quad_boxes.shape[0] is 0:
        continue

    quad_boxes /= input_scale
    quad_boxes *= ([[width, height]] * 4)
    quad_boxes = quad_boxes.astype(np.int32)
    #print(quad_boxes, labels, scores)

    if args.save_img:
        # draw predict points
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = np.array(img, dtype=np.uint8)
        
        # draw gt points
        gt_anno = open(img_dir + "gt/gt_%s.txt" % (_img[:-4]), "r")
        gt_anno = gt_anno.readlines()
        #print(gt_anno)

        for label in gt_anno:
            if args.dataset in ["ICDAR2015"]:
                _x0, _y0, _x1, _y1,_x2, _y2, _x3, _y3, txt = label.split(",")[:9]
                color = (0, 255, 0)
                if "###" in txt:
                    color = (0, 255, 255)

                try:
                    _x0 = int(_x0)
                except:
                    _x0 = int(_x0[1:])

                gt_point = np.array([_x0, _y0, _x1, _y1,_x2, _y2, _x3, _y3], dtype=np.int32)
                gt_point = gt_point.reshape(-1, 4, 2)
                img = cv2.polylines(img, [gt_point], True, color, 2)
            else:
                _xmin, _ymin, _xmax, _ymax = label.split(",")[:4]
                img = cv2.rectangle(img, (int(_xmin), int(_ymin)), (int(_xmax), int(_ymax)), (0,255,0), 2)

        if args.dataset in ["ICDAR2015"]:
            img = cv2.polylines(img, quad_boxes, True, (0,0,255), 2)
        else:
            for quad in quad_boxes:
                xmin = np.min(quad[:, 0])
                ymin = np.min(quad[:, 1])
                xmax = np.max(quad[:, 0])
                ymax = np.max(quad[:, 1])
                img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255), 2)

        save_img_dir = "./icdar2015_test_result/"
        if not os.path.exists(save_img_dir):
            os.mkdir(save_img_dir)
        img_save_path = os.path.join(save_img_dir,_img)
        #print(img_save_path)
        cv2.imwrite(img_save_path, img)

    for quad in quad_boxes:
        if args.dataset in ["ICDAR2015"]:
            [x0, y0], [x1, y1], [x2, y2], [x3, y3] = quad
            f.write("%d,%d,%d,%d,%d,%d,%d,%d\n" % (x0, y0, x1, y1, x2, y2, x3, y3))

        else:
            xmin = np.min(quad[:, 0])
            ymin = np.min(quad[:, 1])
            xmax = np.max(quad[:, 0])
            ymax = np.max(quad[:, 1])
            f.write("%d,%d,%d,%d\n" % (xmin, ymin, xmax, ymax))

    f.close()
    result_zip.write(args.output_zip + "/" + save_file + '.zip', save_file, compress_type=zipfile.ZIP_DEFLATED)
    os.remove(args.output_zip + "/res_%s.txt" % (_img[:-4]))

result_zip.close()

import subprocess

#query = "python %sscript.py -g=%sgt.zip -s=%s" % (eval_dir, eval_dir, eval_dir+args.output_zip)
# return value
#subprocess.call(query, shell=True)
# scorestring = subprocess.check_output(query, shell=True)
# delete the zip file
#os.remove(eval_dir+args.output_zip)
# delete the txt dir
#subprocess.call("rm -rf " + args.output_zip, shell=True)

print("\n\n========== result [ %s ] ==== / option [ input_size=%d, cls_thresh=%.2f, nms_thresh=%.2f======\n" % (FLAGS.tune_from, FLAGS.input_size, FLAGS.cls_thresh, FLAGS.nms_thresh ))