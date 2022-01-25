import os
import cv2
import time
import argparse
from collections import OrderedDict

import torch
import torch.optim as optim
from PIL import Image
from torchvision.transforms import transforms

from craft import CRAFT
from loss.mseloss import Maploss
from torch.autograd import Variable
from data_loader import ICDAR2015
from data.dataset import SynthTextDataLoader
from data.load_icdar import load_icdar2015_gt, load_icdar2013_gt
from utils.inference_boxes import test_net
from data import imgproc
from metrics.eval_det_iou import DetectionIoUEvaluator

import warnings
warnings.filterwarnings(action='ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='CRAFT Finetuning')
    
    parser.add_argument('--Synth_Dir', default='./SynthText', type=str,
                        help='SynthText dataset Directory path')
    parser.add_argument('--ICDAR_Dir', default='./ko_train_data', type=str,help='ICDAR15 dataset Directory path')
    parser.add_argument('--checkpoint', default='./checkpoints/20211129_173635/weights_60000.pth', type=str,
                        help='Checkpoint state_dict file to first training from')
    parser.add_argument('--save_path', default=None, type=str,help='Trained Model Path')
    parser.add_argument('--batch_size', default=18, type = int,
                        help='batch size of training 1:5 ratio')
    parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for Adam')
    parser.add_argument('--gamma', default=0.8, type=float,
                        help='Gamma update for Adam')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--max_iter', default=50000, type=int,
                        help='Number of iteration of training')
    parser.add_argument('--save_interval', type=int, default=500)


    args = parser.parse_args()
    return args


def adjust_learning_rate(optimizer, gamma, step, lr):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = lr * (gamma ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return param_group['lr']

def val(model,evaluator) :
    test_folder = '/data/ICDAR/ICDAR2015'
    text_threshold,low_text,link_threshold,canvas_size,poly,cuda,mag_ratio = 0.7,0.4,0.4,1280,False,True,1.5
    total_imgs_bboxes_gt, total_img_path = load_icdar2015_gt(dataFolder=test_folder, isTraing=False)
    total_img_bboxes_pre = []
    for img_path in total_img_path:
        image = imgproc.loadImage(img_path)
        single_img_bbox = []
        bboxes, polys, score_text = test_net(model,
                                             image,
                                             text_threshold,
                                             link_threshold,
                                             low_text,
                                             cuda,
                                             poly,
                                             canvas_size,
                                             mag_ratio)
        for box in bboxes:
            box_info = {"points": None, "text": None, "ignore": None}
            box_info["points"] = box
            box_info["text"] = "###"
            box_info["ignore"] = False
            single_img_bbox.append(box_info)
        total_img_bboxes_pre.append(single_img_bbox)
    results = []
    for gt, pred in zip(total_imgs_bboxes_gt, total_img_bboxes_pre):
        results.append(evaluator.evaluate_image(gt, pred))
    metrics = evaluator.combine_results(results)
    print(metrics)


def train(Synth_Dir,ICDAR_Dir,checkpoint,save_path,batch_size,lr,num_workers,
          weight_decay,gamma,max_iter,save_interval) :
    synthData_dir = {"synthtext":Synth_Dir}
    Data_dir = ICDAR_Dir
    target_size = 768
    synth_batch = (batch_size//6)
    ICDAR_batch = batch_size - synth_batch
    evaluator = DetectionIoUEvaluator()

    torch.multiprocessing.set_start_method('spawn')

    print('Load the checkpoint of CRAFT {}'.format(checkpoint))
    craft = CRAFT()
    craft = torch.nn.DataParallel(craft).cuda()
    craft.load_state_dict(torch.load(checkpoint))
    print("Load ICDAR dataset")
    ICDAR15DataLoader = ICDAR2015(craft, Data_dir, target_size)
    train_loader = torch.utils.data.DataLoader(ICDAR15DataLoader,
                                               batch_size=ICDAR_batch,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               drop_last=True,
                                               pin_memory=True)

    
    print("Load SynthText dataset")
    synthDataLoader = SynthTextDataLoader(target_size, synthData_dir)
    syn_train_loader = torch.utils.data.DataLoader(synthDataLoader,
                                               batch_size=synth_batch,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               drop_last=True,
                                               pin_memory=True)

    optimizer = optim.Adam(craft.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = Maploss()

    update_lr_rate_step = 2

    train_step = 0
    loss_value = 0
    batch_time = 0
    training_lr = lr 
    #save_path = 'checkpoints/20211202_174139_IC15'
    save_path = 'checkpoints/{}_IC15'.format(time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time())))
    print('New save path: {}'.format(save_path))
    print('Start training...!')
    batch_syn = iter(syn_train_loader)

    while train_step < max_iter:
        for index, (image, region_image, affinity_image, confidence_mask, confidences) in enumerate(train_loader):
            start_time = time.time()
            craft.train()
            if train_step > 0 and train_step % 10000 == 0:
                training_lr = adjust_learning_rate(optimizer, gamma, update_lr_rate_step, lr)
                update_lr_rate_step += 1
            
            syn_image, syn_region_image, syn_affinity_image, syn_confidence_mask,_ = next(batch_syn)
            image = torch.cat((syn_image,image),0)
            image = Variable(image).cuda()
            region_image = torch.cat((syn_region_image,region_image),0)
            region_image_label = Variable(region_image).cuda()
            affinity_image = torch.cat((syn_affinity_image,affinity_image),0)
            affinity_image_label = Variable(affinity_image).cuda()
            confidence_mask = torch.cat((syn_confidence_mask,confidence_mask),0)
            confidence_mask_label = Variable(confidence_mask).cuda()

            output, _ = craft(image)

            out1 = output[:, :, :, 0].cuda()
            out2 = output[:, :, :, 1].cuda()
            loss = criterion(region_image_label, affinity_image_label, out1, out2, confidence_mask_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            end_time = time.time()
            loss_value += loss.item()
            batch_time += (end_time - start_time)
            if train_step % 100 == 0:
                mean_loss = loss_value / 100
                loss_value = 0
                display_batch_time = time.time()
                avg_batch_time = batch_time/100
                batch_time = 0
                print("{}, training_step: {}|{}, learning rate: {:.8f}, training_loss: {:.5f}, avg_batch_time: {:.5f}".format(time.strftime('%H:%M:%S',time.localtime(time.time())), train_step, max_iter, training_lr, mean_loss, avg_batch_time))

            train_step += 1
            craft.eval()
            if train_step % save_interval == 0 and train_step != 0:
                val(craft,evaluator)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                
                print('Saving state, index:', train_step)
                pth_name = os.path.join(save_path, 'weights_' + repr(train_step) + '.pth')
                torch.save(craft.state_dict(),pth_name )

def main(args) :
    train(**args.__dict__)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)

