from terminaltables import AsciiTable

import os, sys, time, datetime, argparse
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch

import tqdm
from data_process.kitti_dataloader import create_train_dataloader, create_val_dataloader

from models.model_utils import create_model, make_data_parallel
from models.darknet2pytorch import Darknet

from utils.evaluation_utils import load_classes
import torch.optim as optim
from eval_mAP import evaluate_mAP

from config.train_config import parse_train_configs

def main():
    # Get data configuration
    configs = parse_train_configs()
    
    print(configs.device)

    # Initiate model
    # model = create_model(configs).to(configs.device)
    model = Darknet(cfgfile=configs.model_def, use_giou_loss=configs.use_giou_loss)
    # model.print_network()
    model = model.to(configs.device)
    
    # Get data configuration
    class_names = load_classes("dataset/classes.names")
    
    print(configs.pretrained_path)
    
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)

    # If specified we start from checkpoint
    if configs.pretrained_path:
        if configs.pretrained_path.endswith(".pth"):
            # model.apply(weights_init_normal)
            # Data Parallel
            # model = make_data_parallel(model, configs)
            model.load_state_dict(torch.load(configs.pretrained_path, map_location="cuda:0"))
            # model.load_state_dict(torch.load(configs.pretrained_path))
            print("Trained pytorch weight loaded!")
        else:
            model.load_darknet_weights(configs.pretrained_path)
            # Data Parallel
            # model = make_data_parallel(model, configs)
            print("Darknet weight loaded!")


    """
    idx_cnt = 0
    for name, param in model.named_parameters():
        # module_list.21.batch_norm_21.bias
        layer_id = int(name.split('.')[1])
        print(idx_cnt,name, layer_id)
        idx_cnt += 1
    idx_cnt = 0
    for param in model.parameters():
        print(idx_cnt, param.requires_grad)
        idx_cnt += 1
    
    idx_cnt = 0
    for name, param in model.named_parameters():
        layer_id = int(name.split('.')[2])
        # print(idx_cnt, layer_id , name)
        if layer_id not in un_freeze_lays:
            # for param in child.parameters():
            param.requires_grad = False
        idx_cnt += 1
    
    idx_cnt = 0
    for param in model.parameters():
        print(idx_cnt, param.requires_grad)
        idx_cnt += 1
    """
    
    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "loss_x",
        "loss_y",
        "loss_w",
        "loss_h",
        "loss_im",
        "loss_re",
        "loss_obj",
        "loss_cls",
        "cls_acc",
        # "recall50",
        # "recall75",
        # "precision",
        "conf_obj",
        "conf_noobj",
    ]
    
    # learning rate scheduler config
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    
    # Create dataloader
    train_dataloader, train_sampler = create_train_dataloader(configs)
    # train_dataloader = create_val_dataloader(configs)
    
    """
    idx_cnt = 0
    for name, param in model.named_parameters():
        layer_id = int(name.split('.')[2])
        # print(idx_cnt, layer_id , name)
        # if layer_id not in un_freeze_lays:
        if layer_id < 137:
            # for param in child.parameters():
            param.requires_grad = False
        idx_cnt += 1
    """

    max_mAP = 0.0
    start_time = time.time() 
    for epoch in range(0, configs.num_epochs, 1):
        
        num_iters_per_epoch = len(train_dataloader)

        print(num_iters_per_epoch)

        # switch to train mode
        model.train()
        
        epoch_loss = 0
        # Training        
        for batch_idx, batch_data in enumerate(tqdm.tqdm(train_dataloader)):
            """
            # print(batch_data)
            
            print(batch_data[0])
            print(batch_data[1])
            print(batch_data[1].shape)
            print(batch_data[2])
            
            imgs = batch_data[1]
            
            from PIL import Image
            import numpy as np

            w, h = imgs[0].shape[1], imgs[0].shape[2]
            src = imgs[0]
            # data = np.zeros((h, w, 3), dtype=np.uint8)
            # data[256, 256] = [255, 0, 0]
            
            data = np.zeros((h, w, 3), dtype=np.uint8)
            data[:,:,0] = src[0,:,:]*255
            data[:,:,1] = src[1,:,:]*255
            data[:,:,2] = src[2,:,:]*255
            # img = Image.fromarray(data, 'RGB')
            img = Image.fromarray(data)
            img.save('my_img.png')
            img.show()
            """
            
            # data_time.update(time.time() - start_time)
            _, imgs, targets = batch_data
            global_step = num_iters_per_epoch * epoch + batch_idx + 1
            
            # batch_size = imgs.size(0)

            targets = targets.to(configs.device, non_blocking=True)
            imgs = imgs.to(configs.device, non_blocking=True)

            total_loss, outputs = model(imgs, targets)
            
            epoch_loss += float(total_loss.item())
            # compute gradient and perform backpropagation
            total_loss.backward()

            if global_step % configs.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                # Adjust learning rate
                lr_scheduler.step()

                # zero the parameter gradients
                optimizer.zero_grad()

            else:
                reduced_loss = total_loss.data
                
            # ----------------
            #   Log progress
            # ----------------
            if (batch_idx+1) % int(len(train_dataloader)/3) == 0:

                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % ((epoch+1), configs.num_epochs, (batch_idx+1), len(train_dataloader))

                metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

                # Log metrics at each YOLO layer
                for i, metric in enumerate(metrics):
                    formats = {m: "%.6f" for m in metrics}
                    formats["grid_size"] = "%2d"
                    formats["cls_acc"] = "%.2f%%"
                    row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                    metric_table += [[metric, *row_metrics]]

                    # Tensorboard logging
                    tensorboard_log = []
                    for j, yolo in enumerate(model.yolo_layers):
                        for name, metric in yolo.metrics.items():
                            if name != "grid_size":
                                tensorboard_log += [(f"{name}_{j+1}", metric)]
                    tensorboard_log += [("loss", total_loss.item())]
                    # logger.list_of_scalars_summary(tensorboard_log, global_step)

                log_str += AsciiTable(metric_table).table
                log_str += f"\nTotal loss {total_loss.item()}"

                # Determine approximate time left for epoch
                epoch_batches_left = len(train_dataloader) - (batch_idx + 1)
                time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_idx + 1))
                log_str += f"\n---- ETA {time_left}"

                print(log_str)

            # model.seen += imgs.size(0)
        crnt_epoch_loss = epoch_loss/num_iters_per_epoch
        
        torch.save(model.state_dict(), configs.save_path)
        # global_epoch += 1
        
        # print("Global_epoch :",global_epoch, "Current epoch loss : {:1.5f}".format(crnt_epoch_loss),'Saved at {}'.format(configs.save_path))
        print("Current epoch loss : {:1.5f}".format(crnt_epoch_loss),'Saved at {}'.format(configs.save_path))
        
    # Evaulation
    print("\n---- Evaluating Model ----")
    val_dataloader = create_val_dataloader(configs)
    precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs)

    val_metrics_dict = {
        'precision': precision.mean(),
        'recall': recall.mean(),
        'AP': AP.mean(),
        'f1': f1.mean(),
        'ap_class': ap_class.mean()
    }

    # Print class APs and mAP
    ap_table = [["Index", "Class name", "AP"]]
    for i, c in enumerate(ap_class):
        ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
    print(AsciiTable(ap_table).table)
    print(f"---- mAP {AP.mean()}")

    max_mAP = AP.mean()
    #-------------------------------------------------------------------------------------
    """
    # Save checkpoint
    if (epoch+1) % configs.checkpoint_freq == 0:
        torch.save(model.state_dict(), configs.save_path)
        print('save a checkpoint at {}'.format(configs.save_path))
    """
            
if __name__ == '__main__':
    main()
    
