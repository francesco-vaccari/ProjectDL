import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from RefcocogDataset import RefcocogDataset
from torch.utils.data import DataLoader

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_sample(sample, bbox, idx=0):
    print(f"Sentence: {sample['sentences'][idx]}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(sample['image'][idx].permute(1, 2, 0))
    axes[1].imshow(bbox['gt'][idx])
    plt.tight_layout()
    plt.show()


def visualize_loss(map, bbox, idx, loss_map):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(map)
    axs[1].imshow(bbox['gt'][idx])
    axs[2].imshow(loss_map.reshape(14, 14))
    plt.tight_layout()
    plt.show()

def load_scheduler(scheduler, path):
    scheduler.load_state_dict(torch.load(path))
    return scheduler

def load_optimizer(optimizer, path):
    optimizer.load_state_dict(torch.load(path))
    return optimizer

class BatchLossFunction(nn.Module):
    def __init__(self, gamma=3.4, average=True):
        super(BatchLossFunction, self).__init__()
        self.gamma = gamma
        self.average = average

    def forward(self, patch_tokens, out_text, gt):
        loss = torch.zeros(1, requires_grad=True)
        for idx in range(patch_tokens.shape[0]):
            pt = patch_tokens[idx, 1:]
            ot = out_text[idx, :].unsqueeze(0)
            map = torch.zeros(196)

            for i, token in enumerate(pt):
                map[i] = 1 - torch.cosine_similarity(token, ot).item() # 1 - ... temporary fix

            vector = torch.sigmoid(map)

            gt_map = gt[idx]/255
            gt_vector = gt_map.reshape(-1)

            abs = torch.abs(vector - gt_vector)
            log = -torch.log(1-abs)

            # amplify the error of pixels that should belong to the object
            log = log*(gt_vector*self.gamma+1)
            loss = loss + torch.sum(log)

        if self.average:
            return (loss / patch_tokens.shape[0])
        else:
            return loss

from datetime import datetime
from tqdm import tqdm
import os

def train_one_epoch(epoch_index, train_loader, model, criterion, optimizer, loop):
    epoch_losses = []
    for i, (samples, bbox) in enumerate(train_loader):
        loop.set_postfix_str(f'Batch {i+1}/{len(train_loader)}')

        optimizer.zero_grad()

        images = samples['image'].to(device)
        sentences = clip.tokenize(samples['sentences']).to(device)

        out_image, out_text, patch_tokens, text_tokens = model.encode(images, sentences)

        batch_loss = criterion(patch_tokens, out_text, bbox['gt'])

        batch_loss.backward()
        optimizer.step()

        epoch_losses.append(batch_loss)

    return torch.mean(torch.tensor(epoch_losses)).item()

def train_loop(num_epochs, train_loader, model, criterion, optimizer, scheduler, eval_loader):
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
                out_image, out_text, patch_tokens, text_tokens = model.encode(images, sentences)
                batch_loss = criterion(patch_tokens, out_text, bbox['gt'])
                eval_losses.append(batch_loss)

            eval_loss = torch.mean(torch.tensor(eval_losses)).item()
            loop.write(f'Epoch {epoch+1}/{num_epochs}\tEval loss: {eval_loss:.4f}')
            wandb.log({"train_loss": epoch_loss, "eval_loss": eval_loss})

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(model.state_dict(), run_path + "/best.pth")
        
        scheduler.step()

        torch.save(model.state_dict(), run_path + "/epoch_" + str(epoch+1) + ".pth")
        torch.save(optimizer.state_dict(), run_path + "/optimizer_epoch_" + str(epoch+1) + ".pth")
        torch.save(scheduler.state_dict(), run_path + "/scheduler_epoch_" + str(epoch+1) + ".pth")



model, preprocess = clip.load("ViT-B/16") # only works with ViT-B/16
model.init_adapters() # needed because state dict of clip does not contain adapters, goes before moving to gpu
# model.load_parameters(path="") # for when we have state dict of adapters trained, goes after adapters init
model = model.to(device)

model.freeze_for_training() # freezes all clip by putting requires_grad=False and then unfreezes adapters

batch_size = 48 # 48 should be possible
train_dataset = RefcocogDataset("./refcocog", split="train", transform=preprocess)
val_dataset = RefcocogDataset("./refcocog", split="val", transform=preprocess)
test_dataset = RefcocogDataset("./refcocog", split="test", transform=preprocess)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

learning_rate = 5e-5
weight_decay = 5e-3
num_epochs = 60 # 60

criterion = BatchLossFunction(gamma=3.4, average=True) # keep 3.4 for now
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
# optimizer = load_optimizer(optimizer, path="") # when needed to resume training
scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=num_epochs)
# scheduler = load_scheduler(scheduler, path="") # when needed to resume training

wandb.init(project="projectdl", 
           name='locator', 
           config={
               "learning_rate": learning_rate,
               "weight_decay": weight_decay,
               "batch_size": batch_size,
               "num_epochs": num_epochs,
               "gamma": 3.4
            }
)

train_loop(num_epochs, train_loader, model, criterion, optimizer, scheduler, val_loader)