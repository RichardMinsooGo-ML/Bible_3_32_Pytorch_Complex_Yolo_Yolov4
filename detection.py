# python detection.py --model_def config/cfg/complex_yolov4.cfg --pretrained_path checkpoints/Complex_yolo_yolo_v4.pth
# python detection.py --model_def config/cfg/complex_yolov4_tiny.cfg.cfg --pretrained_path checkpoints/Complex_yolo_yolo_v3_tiny.pth --batch_size 8  
import os, sys, time, datetime, argparse
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch

import config.kitti_config as cnf
import cv2
from data_process import kitti_utils, kitti_bev_utils
from data_process.kitti_dataloader import create_test_dataloader
from models.model_utils import create_model, make_data_parallel

from utils.evaluation_utils import post_processing, rescale_boxes, post_processing_v2
from utils.misc import time_synchronized
from utils.mayavi_viewer import show_image_with_boxes, merge_rgb_to_bev, predictions_to_kitti_format

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-a", "--arch", type=str, default="darknet", metavar="ARCH", help="The name of the model architecture")
    
    parser.add_argument("--model_def", type=str, default="config/cfg/complex_yolov4.cfg", metavar="PATH", help="The path for cfgfile (only for darknet)")
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/Complex_yolo_yolo_v4.pth", metavar="PATH", help="the path of the pretrained checkpoint")
    parser.add_argument("--batch_size"  , type=int  , default=1, help="size of each image batch")
    parser.add_argument("--conf_thresh"  , type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thresh", type=float, default=0.5, help="the threshold for conf")

    parser.add_argument("--img_size", type=int, default=608, help="the size of input image")
    parser.add_argument("--use_giou_loss", action="store_true", help="If true, use GIoU loss during training. If false, use MSE loss for training")

    parser.add_argument("--gpu_idx", default=None, type=int, help="GPU index to use.")
    parser.add_argument("--num_samples", type=int, default=None, help="Take a subset of the dataset to run and debug")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of threads for loading data")

    parser.add_argument("--show_image", action="store_true", help="If true, show the image during demostration")
    
    parser.add_argument("--save_test_output", type=bool, default=True, help="If true, the output image of the testing phase will be saved")
    parser.add_argument("--output_format", type=str, default="video", metavar="PATH", help="the type of the test output (support image or video)")
    parser.add_argument("--output_video_fn", type=str, default="pred_complex_yolo_v4", metavar="PATH", help="the video filename if the output format is video")

    configs = parser.parse_args()
    
    configs.pin_memory = True

    configs.dataset_dir = os.path.join("dataset", "kitti")

    if configs.save_test_output:
        configs.results_dir = "pred_IMAGES"
        if not os.path.exists(configs.results_dir):
            os.makedirs(configs.results_dir)

    configs.distributed = False  # For testing

    print(configs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(configs)
    
    
    # model.print_network()
    print("\n" + "___m__@@__m___" * 10 + "\n")
    
    print(configs.pretrained_path)    
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    
    model = model.to(device)

    # Load checkpoint weights
    if configs.pretrained_path:
        if configs.pretrained_path.endswith(".pth"):
            model.load_state_dict(torch.load(configs.pretrained_path))
            # model.load_state_dict(torch.load(configs.pretrained_path,map_location='cuda:0'))
            print("Trained pytorch weight loaded!")
    
    # Data Parallel
    model = make_data_parallel(model, configs)
    
    out_cap = None
    # Eval mode
    model.eval()

    test_dataloader = create_test_dataloader(configs)
    
    for batch_idx, (img_paths, imgs_bev) in enumerate(test_dataloader):
        input_imgs = imgs_bev.to(device).float()
        t1 = time_synchronized()
        outputs = model(input_imgs)
        t2 = time_synchronized()
        
        with torch.no_grad():
            detections = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)

        img_detections = []  # Stores detections for each image index
        img_detections.extend(detections)

        img_bev = imgs_bev.squeeze() * 255
        img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
        img_bev = cv2.resize(img_bev, (configs.img_size, configs.img_size))
        
        
        for detections in img_detections:
            if detections is None:
                continue

            # Rescale boxes to original image
            detections = rescale_boxes(detections, configs.img_size, img_bev.shape[:2])
            for x, y, w, l, im, re, *_, cls_pred in detections:
                yaw = np.arctan2(im, re)
                # Draw rotated box
                kitti_bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(cls_pred)])

        img_rgb = cv2.imread(img_paths[0])
        calib = kitti_utils.Calibration(img_paths[0].replace(".png", ".txt").replace("image_2", "calib"))
        objects_pred = predictions_to_kitti_format(img_detections, calib, img_rgb.shape, configs.img_size)
        img_rgb = show_image_with_boxes(img_rgb, objects_pred, calib, False)
        
        img_bev = cv2.flip(cv2.flip(img_bev, 0), 1)

        out_img = merge_rgb_to_bev(img_rgb, img_bev, output_width=608)

        print("\tDone testing the {}th sample, time: {:.1f}ms, speed {:.2f}FPS".format(batch_idx+1, (t2 - t1) * 1000,
                                                                                       1 / (t2 - t1)))

        if configs.save_test_output:
            if configs.output_format == "image":
                img_fn = os.path.basename(img_paths[0])[:-4]
                cv2.imwrite(os.path.join(configs.results_dir, "{}.jpg".format(img_fn)), out_img)
            elif configs.output_format == "video":
                if out_cap is None:
                    out_cap_h, out_cap_w = out_img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    out_cap = cv2.VideoWriter(
                        os.path.join(configs.results_dir, "{}.avi".format(configs.output_video_fn)),
                        fourcc, 10, (out_cap_w, out_cap_h))

                out_cap.write(out_img)
            else:
                raise TypeError

        configs.show_image = True
        
        if configs.show_image:
            cv2.imshow("test-img", out_img)
            # print("\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n")
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()
            
if __name__ == '__main__':
    main()
