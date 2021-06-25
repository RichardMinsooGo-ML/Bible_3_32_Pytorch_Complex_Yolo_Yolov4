
import os, sys, time, datetime, argparse
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
from data_process.kitti_dataloader import create_val_dataloader
import torch.utils.data.distributed
import tqdm

sys.path.append("./")

from models.model_utils import create_model, make_data_parallel

# from utils.misc import AverageMeter, ProgressMeter
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
            # Rescale x, y, w, h of targets ((box_idx, class, x, y, w, l, im, re))
            targets[:, 2:6] *= configs.img_size
            imgs = imgs.to(configs.device, non_blocking=True)

            outputs = model(imgs)
            outputs = post_processing_v2(outputs, conf_thresh=configs.conf_thres, nms_thresh=configs.nms_thres)

            sample_metrics += get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold=configs.iou_thres)

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--arch", type=str, default="darknet", metavar="ARCH", help="The name of the model architecture")
        
    # parser.add_argument("--model_def", type=str, default="config/cfg/complex_yolov4.cfg", metavar="PATH",help="path to model definition file")
    # parser.add_argument("--pretrained_path", type=str, default="checkpoints/complex_yolov4_mse_loss.pth", metavar="PATH", help="path to weights file")
    parser.add_argument("--model_def", type=str, default="config/cfg/complex_yolov4.cfg", metavar="PATH",help="path to model definition file")
    parser.add_argument("--pretrained_path", type=str, default="checkpoints/complex_yolov4_mse_loss.pth", metavar="PATH", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="dataset/kitti/classes_names.txt", metavar="PATH", help="path to class label file")
    parser.add_argument("--batch_size"  , type=int  , default=4, help="size of each image batch")
    parser.add_argument("--iou_thres"   , type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres",  type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size",   type=int,   default=608, help="size of each image dimension")

    parser.add_argument("--save_path", type=str, default="checkpoints/Model_complexer_yolo_V4_v1.pth", metavar="PATH", help="path to weights file")
    
    parser.add_argument("--use_giou_loss", action="store_true", help="If true, use GIoU loss during training. If false, use MSE loss for training")
    parser.add_argument("--gpu_idx", default=None, type=int, help="GPU index to use.")
    parser.add_argument("--num_samples", type=int, default=None, help="Take a subset of the dataset to run and debug")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of threads for loading data")
    
    configs = parser.parse_args()
    
    configs.pin_memory = True

    configs.dataset_dir = os.path.join("dataset", "kitti")
    
    configs.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initiate model
    model = create_model(configs)
    
    # Get data configuration
    classes = load_classes(configs.class_path)
    
    # model.print_network()
    print("\n" + "___m__@@__m___" * 10 + "\n")
    
    print(configs.pretrained_path)    
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    
    model = model.to(device = configs.device)
    
    # Load checkpoint weights
    if configs.pretrained_path:
        if configs.pretrained_path.endswith(".pth"):
            # model.load_state_dict(torch.load(configs.pretrained_path))
            model.load_state_dict(torch.load(configs.pretrained_path,map_location='cuda:0'))
            print("Trained pytorch weight loaded!")
    
    # Data Parallel
    model = make_data_parallel(model, configs)
    
    torch.save(model.state_dict(), configs.save_path)
    
    # Eval mode
    model.eval()
    
    print("Create the validation dataloader")
    val_dataloader = create_val_dataloader(configs)

    print("\nStart computing mAP...\n")
    precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs)

    print("\nDone computing mAP...\n")
    for idx, cls in enumerate(ap_class):
        print("\t>>>\t Class {} ({}): precision = {:.4f}, recall = {:.4f}, AP = {:.4f}, f1: {:.4f}".format(cls, \
                classes[cls][:3], precision[idx], recall[idx], AP[idx], f1[idx]))

    print("\nmAP: {:.4f}\n".format(AP.mean()))
            
if __name__ == '__main__':
    main()
    