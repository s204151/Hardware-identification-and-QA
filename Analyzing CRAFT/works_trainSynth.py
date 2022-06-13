import os, psutil
import torch
import torch.optim as optim
import cv2
import time
import argparse

from my_dataset import QaDataLoader, watershed, copyStateDict
import wandb
from craft import CRAFT
from loss.mseloss import Maploss
from torch.autograd import Variable
import torch.multiprocessing as mp
import numpy as np
torch.backends.cudnn.deterministic=True

#torch.backends.cudnn.benchmark=True

import random
the_manual_seed =42
torch.manual_seed(the_manual_seed)
random.seed(the_manual_seed)
#Pull_item returns: image, region_image, affinity_image, confidence_mask, confidences
#so data is a tuple of lists containing images, region images... etc,
#

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def collate_tokenize(data):
    images, region_scores, affinity_scores, confidence_masks, confidences = zip(*data)

    try:
        images, region_scores, affinity_scores, confidence_masks = np.array(images,dtype=
                                                                        'float32'),np.array(region_scores, dtype='float32'),np.array(affinity_scores, dtype='float32'),np.array(confidence_masks, dtype='float32')
        images, region_scores, affinity_scores, confidence_masks = torch.tensor(images), torch.tensor(
            region_scores), torch.tensor(affinity_scores), torch.tensor(confidence_masks)

    except ValueError:
        print("Nope! images, region scores, aff or conf masks has bad shape")


    max_len = 0
    obs = len(confidences)
    for i in range(obs):
        conf_len = len(confidences[i])
        if conf_len > max_len:
            max_len = conf_len

    np_confidences = np.array(confidences,dtype = 'object')

    for j in range(obs):
        k = len(confidences[j])
        np_confidences[j] = np.pad(np_confidences[j], (0,max_len - k))

    #print(images.shape)
    try:
        confidences = torch.tensor(np.array(np.reshape(np.concatenate(np_confidences,axis = 0),(obs, max_len)),dtype = 'float32'))
    except ValueError:
        print("np_conf is bad!")

    return images, region_scores, affinity_scores, confidence_masks, confidences


def adjust_learning_rate(optimizer, gamma, step, lr):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = lr * (gamma ** step)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return param_group['lr']


if __name__ == "__main__":

    mp.set_start_method('spawn', force=True)
    #synthData_dir = {"synthtext":"/data/CRAFT-Reimplementation/dataset/SynthText"}
    target_size = 768
    batch_size = 8
    num_workers = 4
    lr = 1e-4
    training_lr = 1e-4
    weight_decay = 5e-4
    gamma = 0.8
    whole_training_step = 10000

    wandb.init(
        project=str(
            f"CRAFT_iter_{whole_training_step}"),
        name=f"experiment_seed{the_manual_seed}",
        config={
            "learning_rate": lr,
            "weight_decay": weight_decay,
            #"rho": rho,
            "lr": training_lr,
            "gamma": gamma,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "num_iter": whole_training_step,
            "target_size": target_size,
            "manualSeed": the_manual_seed,
        })
    args = parser.parse_args()

    craft = CRAFT(freeze=True).cuda()
    craft.load_state_dict(copyStateDict(
        torch.load('/zhome/a7/0/155527/Desktop/s204161/craft_mlt_25k.pth', map_location=torch.device('cuda'))))
    #craft = torch.nn.DataParallel(craft)

    #synthDataLoader = QaDataLoader(target_size=target_size,synthData_dir=synthData_dir...)

    QADataL = QaDataLoader(net=craft, target_size=target_size)

    train_loader = torch.utils.data.DataLoader(QADataL,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               drop_last=True,
                                               pin_memory=True,
                                               collate_fn = collate_tokenize)


    #
    # craft = CRAFT()
    # craft = torch.nn.DataParallel(craft).cuda()
    # craft.load_state_dict(torch.load("/data/CRAFT-Reimplementation/dataset/weights_7000.pth"))

    optimizer = optim.Adam(craft.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = Maploss()

    update_lr_rate_step = 2

    train_step = 0
    loss_value = 0
    batch_time = 0
    while train_step < whole_training_step:

        for index, (image, region_image, affinity_image, confidence_mask, confidences) in enumerate(train_loader):
            print(f"{psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} megabytes memory in usage")
            start_time = time.time()
            craft.train()
            if train_step>0 and train_step%20000==0:
                training_lr = adjust_learning_rate(optimizer, gamma, update_lr_rate_step, lr)
                update_lr_rate_step += 1

            images = Variable(image).cuda()
            region_image_label = Variable(region_image).cuda()
            affinity_image_label = Variable(affinity_image).cuda()
            confidence_mask_label = Variable(confidence_mask).cuda()

            output, _ = craft(images)

            out1 = output[:, :, :, 0]
            out2 = output[:, :, :, 1]
            loss = criterion(region_image_label, affinity_image_label, out1, out2, confidence_mask_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            end_time = time.time()
            loss_value += loss.item()
            batch_time += (end_time - start_time)
            if train_step > 0 and train_step%5==0:
                mean_loss = loss_value / 5
                loss_value = 0
                wandb.log({"mean train loss á 5th iteration": mean_loss})
                display_batch_time = time.time()
                avg_batch_time = batch_time/5
                batch_time = 0
                print("{}, training_step: {}|{}, learning rate: {:.8f}, training_loss: {:.5f}, avg_batch_time: {:.5f}".format(time.strftime('%Y-%m-%d:%H:%M:%S',time.localtime(time.time())), train_step, whole_training_step, training_lr, mean_loss, avg_batch_time))

            train_step += 1
            #print(train_step)

            if train_step % 1000 == 0 and train_step != 0:
                print('Saving state, index:', train_step)
                torch.save(craft.state_dict(),
                           '/zhome/a7/0/155527/Desktop/s204161/fagprojekt/CRAFT-Reimplementation-craft/saved_weights/10kweights_' + repr(train_step) + '.pth')
                # test('/data/CRAFT-pytorch/synweights/synweights_' + repr(index) + '.pth')
                #test('/data/CRAFT-pytorch/craft_mlt_25k.pth')
                # getresult()

