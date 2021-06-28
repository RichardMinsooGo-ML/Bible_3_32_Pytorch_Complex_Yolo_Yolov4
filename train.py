# python train.py --model_def config/cfg/yolo3d_yolov4.cfg --pretrained_path checkpoints/Model_yolo3d_yolov4.pth --save_path checkpoints/Model_yolo3d_yolov4.pth
# python train.py --model_def config/cfg/yolo3d_yolov4.cfg --pretrained_path checkpoints/yolov4.weights --save_path checkpoints/Model_yolo3d_yolov4.pth

# python train.py  --model_def config/cfg/yolo3d_yolov4_tiny.cfg --pretrained_path checkpoints/Model_yolo3d_yolov4_tiny.pth --save_path checkpoints/Model_yolo3d_yolov4_tiny.pth
# python train.py  --model_def config/cfg/yolo3d_yolov4_tiny.cfg --pretrained_path checkpoints/yolov4-tiny.weights --save_path checkpoints/Model_yolo3d_yolov4_tiny.pth

from terminaltables import AsciiTable

import os, sys, time, datetime, argparse
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch

import tqdm
from data_process.kitti_dataloader import create_train_dataloader, create_val_dataloader

from models.model_utils import create_model, make_data_parallel

from utils.evaluation_utils import load_classes
from config.train_config import parse_train_configs
import torch.optim as optim
from eval_mAP import evaluate_mAP

import pickle

def main():
    
    configs = parse_train_configs()

    # Get data configuration
    configs.device = torch.device('cpu' if configs.gpu_idx is None else 'cuda:{}'.format(configs.gpu_idx))
    print(configs.device)
    # Initiate model
    model = create_model(configs).to(configs.device)
    # model.print_network()
    
    # Get data configuration
    class_names = load_classes("dataset/classes.names")

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

    optimizer = torch.optim.Adam(model.parameters())

    # learning rate scheduler config
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    
    # global_epoch = 0
    
    f = open('checkpoints/global_epoch.pckl', 'rb')
    global_epoch = pickle.load(f)
    f.close()
    
    # train_dataloader, train_sampler = create_train_dataloader(configs)
    train_dataloader = create_val_dataloader(configs)
    
    max_mAP = 0.5
        
    for epoch in range(0, configs.num_epochs, 1):
            
        num_iters_per_epoch = len(train_dataloader)        

        # print(num_iters_per_epoch)

        # switch to train mode
        model.train()
        # start_time = time.time()
        
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

            import sys
            sys.exit()
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
            
        crnt_epoch_loss = epoch_loss/num_iters_per_epoch
        # Evaulation        
        
        torch.save(model.state_dict(), configs.save_path)
        global_epoch += 1
        
        print("Global_epoch :",global_epoch, "/ loss : {:1.5f}".format(crnt_epoch_loss),'Saved at {}'.format(configs.save_path))
        
        f = open('checkpoints/global_epoch.pckl', 'wb')
        pickle.dump(global_epoch, f)
        f.close()
        
        """
        if min_epoch_loss > crnt_epoch_loss: 
            min_epoch_loss = crnt_epoch_loss

            f = open('checkpoints/min_epoch_loss.pckl', 'wb')
            pickle.dump(min_epoch_loss, f)
            f.close()
        """
        
        """
        if trial_cnt < trial_epoch:
            
            trial_cnt += 1
            # Save checkpoint
            if min_epoch_loss > crnt_epoch_loss:
                torch.save(model.state_dict(), configs.save_path)
                print('save a checkpoint at {}'.format(configs.save_path))
                min_epoch_loss = crnt_epoch_loss

                f = open('checkpoints/min_epoch_loss.pckl', 'wb')
                pickle.dump(min_epoch_loss, f)

                print("min_epoch_loss :",min_epoch_loss)
                trial_cnt = 0

            # Evaulation        
            #-------------------------------------------------------------------------------------        
            # if (epoch+1) % 5 == 0 and (epoch+1) >= 2:
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

            else:
                print("Epoch_loss :",crnt_epoch_loss)

                if trial_cnt == trial_epoch:
                    model.load_state_dict(torch.load(configs.pretrained_path))
                    f = open('checkpoints/min_epoch_loss.pckl', 'rb')
                    min_epoch_loss = pickle.load(f)

                    print("min_epoch_loss weight will be used again!")
                
                    trial_cnt = 0
                    
        print("trial_cnt :", trial_cnt)
        """            
        
    # Evaulation        
    #-------------------------------------------------------------------------------------        
    # if (epoch+1) % 8 == 0:

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

if __name__ == '__main__':
    main()
    