import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from RefcocogDataset import RefcocogDataset
from torch.utils.data import DataLoader

import torchvision.ops.focal_loss as focal_loss
# import wandb
from datetime import datetime
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
torch.autograd.set_detect_anomaly(True)

logwandb = False

def load_scheduler(scheduler, path):
    scheduler.load_state_dict(torch.load(path))
    return scheduler

def load_optimizer(optimizer, path):
    optimizer.load_state_dict(torch.load(path))
    return optimizer

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # inputs = torch.sigmoid(inputs)
        
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        
        # print("### DEBUG LOSS FUNCTION")
        # print(inputs)
        # print(targets)
        # print(targets.sum())
        
        # intersection = (inputs * targets).sum()                            
        # print(intersection)
        # dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        # print(dice)
        # print("### END --- DEBUG LOSS FUNCTION")

        # return 1 - dice
        smooth = 1.

        # have to use contiguous since they may from a torch.view op
        iflat = inputs.contiguous().view(-1)
        tflat = targets.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(tflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )
        
    
class CustomLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.diceLoss = DiceLoss()

    def forward(self, inputs, targets):
        f_loss = focal_loss.sigmoid_focal_loss(inputs, targets, alpha=self.alpha, gamma=self.gamma, reduction="mean")
        d_loss = self.diceLoss(inputs, targets)
        loss = 1.75*f_loss + 1*d_loss
        return loss


def train_one_epoch(epoch_index, train_loader, model, criterion, optimizer, loop):
    epoch_losses = []
    for i, (samples, bbox) in enumerate(train_loader):
        loop.set_postfix_str(f'Batch {i+1}/{len(train_loader)}')


        images = samples['image'].to(device)
        sentences = clip.tokenize(samples['sentences']).to(device)
        target = bbox['gt'].to(device, dtype=torch.float32)

        optimizer.zero_grad()

        maps, fv = model.encode(images, sentences)

        batch_loss = criterion(maps, target)

        batch_loss.backward()
        optimizer.step()

        for param in model.backbone_adapters_MLP_vis[0].up_proj.parameters():
            print(param)

        epoch_losses.append(batch_loss)
        if logwandb:
            wandb.log({"batch_loss": batch_loss.item()})

    return torch.mean(torch.tensor(epoch_losses)).item()

def train_loop(num_epochs, train_loader, model, criterion, optimizer, scheduler, eval_loader, num_epochs_trained=0):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_path = 'runs/run_{}'.format(timestamp)
    cmd = f'mkdir runs; mkdir runs/run_{timestamp}'
    os.system(cmd)

    best_eval_loss = float('inf')

    loop = tqdm(range(num_epochs), desc="Training locator", leave=True)
    for epoch in loop:
        model.train()
        epoch_loss = train_one_epoch(epoch, train_loader, model, criterion, optimizer, loop)

        model.eval()
        eval_losses = []
        with torch.no_grad():
            for samples, bbox in eval_loader:
                images = samples['image'].to(device)
                sentences = clip.tokenize(samples['sentences']).to(device)
                maps, fv = model.encode(images, sentences)

                batch_loss = criterion(maps, bbox['gt'].to(device, dtype=torch.float32))

                eval_losses.append(batch_loss)

            eval_loss = torch.mean(torch.tensor(eval_losses)).item()
            loop.write(f'Epoch {epoch+1}/{num_epochs}\tEval loss: {eval_loss:.4f}')
            if logwandb:
                wandb.log({"train_loss": epoch_loss, "eval_loss": eval_loss})

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(model.cpu().state_dict(), run_path + "/best.pth")
        
        scheduler.step()

        torch.save(model.cpu().state_dict(), run_path + "/epoch_" + str(epoch+num_epochs_trained+1) + ".pth")
        torch.save(optimizer.cpu().state_dict(), run_path + "/optimizer_epoch_" + str(epoch+num_epochs_trained+1) + ".pth")
        torch.save(scheduler.cpu().state_dict(), run_path + "/scheduler_epoch_" + str(epoch+num_epochs_trained+1) + ".pth")


if __name__ == "__main__":
    ########################################
    # LOAD CLIP MODEL
    ########################################

    model, preprocess = clip.load("ViT-B/16") # only works with ViT-B/16


    ########################################
    # INITIALIZE DATASET
    ########################################

    batch_size = 32 # 48 should be possible
    train_dataset = RefcocogDataset("../Dataset/refcocog", split="train", transform=preprocess)
    val_dataset = RefcocogDataset("../Dataset/refcocog", split="val", transform=preprocess)
    test_dataset = RefcocogDataset("../Dataset/refcocog", split="test", transform=preprocess)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ########################################
    # INITIALIZE TRAINING PARAMETERS
    ########################################

    learning_rate = 5e-5 # 5e-5
    weight_decay = 5e-3 # 5e-3
    num_epochs = 60 # change if epochs alredy trained
    num_epochs_trained = 0 # change if epochs alredy trained


    ########################################
    # INITIALIZE MODEL
    ########################################

    # model.init_adapters() # adds adapters after original state dict has been loaded
    # model.load_parameters(path="") # when needed to resume training
    model.freeze_for_training() # freezes all clip by putting requires_grad=False and then unfreezes adapters

    model = model.to(device)


    ########################################
    # INITIALIZE LOSS FUNCTION, OPTIMIZER AND SCHEDULER
    ########################################

    criterion = CustomLoss(alpha=0.65, gamma=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay, eps=1e-04)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    # optimizer = load_optimizer(optimizer, path="") # when needed to resume training
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=num_epochs)
    # scheduler = load_scheduler(scheduler, path="") # when needed to resume training

    if logwandb:
        wandb.init(project="projectdl", 
                name='locator5', 
                config={
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                        "num_epochs_trained": num_epochs_trained,
                    "loss_fn": "1.75*focal+dice loss"
                    }
        )

    ########################################
    # TRAINING LOOP
    ########################################

    train_loop(num_epochs, train_loader, model, criterion, optimizer, scheduler, val_loader, num_epochs_trained)

    wandb.finish()