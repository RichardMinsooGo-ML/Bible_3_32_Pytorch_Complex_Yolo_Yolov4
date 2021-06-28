# python eval_mAP.py --model_def config/cfg/yolo3d_yolov4.cfg --pretrained_path checkpoints/Model_yolo3d_yolov4.pth
# python eval_mAP.py --model_def config/cfg/yolo3d_yolov4_tiny.cfg --pretrained_path checkpoints/Model_yolo3d_yolov4_tiny.pth

import os, sys, time, datetime, argparse
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import torch.utils.data.distributed
import tqdm
sys.path.append("./")

from data_process.kitti_dataloader import create_val_dataloader

from models.model_utils import create_model, make_data_parallel

from utils.evaluation_utils import get_batch_statistics_rotated_bbox, ap_per_class, load_classes, post_processing_v2

def evaluate_mAP(val_loader, model, configs):
    # switch to evaluate mode
    model.eval()

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    with torch.no_grad():
        # start_time = time.time()
        for batch_idx, batch_data in enumerate(tqdm.tqdm(val_loader)):
            # data_time.update(time.time() - start_time)
            _, imgs, targets = batch_data
            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale x, y, w, h of targets ((box_idx, class, x, y, z, h, w, l, im, re))
            targets[:, 2:4] *= configs.img_size
            targets[:, 5:8] *= configs.img_size
            imgs = imgs.to(configs.device, non_blocking=True)

            outputs = model(imgs)
            outputs = post_processing_v2(outputs, conf_thresh=configs.conf_thres, nms_thresh=configs.nms_thres)

            sample_metrics += get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold=configs.iou_thres)

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_def", type=str, default="config/cfg/yolo3d_yolov4.cfg", metavar="PATH", help="The path for cfgfile (only for darknet)")
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/Model_yolo3d_yolov4.pth", metavar="PATH", help="the path of the pretrained checkpoint")
    
    # parser.add_argument("--model_def", type=str, default="config/cfg/yolo3d_yolov4_tiny.cfg", metavar="PATH", help="The path for cfgfile (only for darknet)")
    # parser.add_argument("--pretrained_path", type=str, default="checkpoints/Model_yolo3d_yolov4_tiny.pth", metavar="PATH", help="the path of the pretrained checkpoint")
    
    parser.add_argument("-a", "--arch", type=str, default="darknet", metavar="ARCH", help="The name of the model architecture")
    parser.add_argument("--batch_size"  , type=int  , default=4, help="size of each image batch")
    
    parser.add_argument("--use_giou_loss", action="store_true", help="If true, use GIoU loss during training. If false, use MSE loss for training")

    parser.add_argument("--no_cuda", action="store_true", help="If true, cuda is not used.")
    parser.add_argument("--gpu_idx", default=None, type=int, help="GPU index to use.")

    parser.add_argument("--num_samples", type=int, default=None, help="Take a subset of the dataset to run and debug")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of threads for loading data")
    parser.add_argument("--class_path", type=str, default="dataset/kitti/classes_names.txt", metavar="PATH", help="The class names of objects in the task")
    parser.add_argument("--iou_thres"   , type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres"  , type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres"   , type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size",   type=int,   default=608, help="the size of input image")

    configs = parser.parse_args()
    configs.pin_memory = True

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.working_dir = "./"
    configs.dataset_dir = os.path.join(configs.working_dir, "dataset", "kitti")

    configs.distributed = False  # For evaluation
    class_names = load_classes(configs.class_path)

    configs.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # configs.device = torch.device("cpu" if configs.no_cuda else "cuda:{}".format(configs.gpu_idx))
    print(configs.device)
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initiate model
    model = create_model(configs).to(configs.device)
    # model.print_network()
    
    # If specified we start from checkpoint
    if configs.pretrained_path:
        if configs.pretrained_path.endswith(".pth"):
            # Data Parallel
            model = make_data_parallel(model, configs)
            model.load_state_dict(torch.load(configs.pretrained_path))
            print("Trained pytorch weight loaded!")
        else:
            model.load_darknet_weights(configs.pretrained_path)
            # Data Parallel
            model = make_data_parallel(model, configs)
            print("Darknet weight loaded!")

    
    # model.print_network()
    print("\n" + "___m__@@__m___" * 10 + "\n")
    
    print(configs.pretrained_path)
    
    model.eval()
        
    print("Create the validation dataloader")
    val_dataloader = create_val_dataloader(configs)

    print("\nStart computing mAP...\n")
    precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs)

    print("\nDone computing mAP...\n")
    for idx, cls in enumerate(ap_class):
        print("\t>>>\t Class {} ({}): precision = {:.4f}, recall = {:.4f}, AP = {:.4f}, f1: {:.4f}".format(cls, \
                class_names[cls][:3], precision[idx], recall[idx], AP[idx], f1[idx]))

    print("\nmAP: {:.4}\n".format(AP.mean()))
