import clip
import torch
import argparse

from dataset.RefcocogDataset import RefcocogDataset
from torch.utils.data import DataLoader

from FocalDiceLoss import FocalDiceLoss
import wandb
from datetime import datetime
from tqdm import tqdm
import os

############################################
# DEFINE ARGUMENTS
############################################

arg = argparse.ArgumentParser()
arg.add_argument("--name", type=str, default='run_{}'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), help="Name of the run")
arg.add_argument("--batch_size", type=int, default=16, help="Batch size")
arg.add_argument("--num_epochs", type=int, default=60, help="Number of epochs")
arg.add_argument("--dataset", type=str, default="../Dataset/refcocog", help="Dataset to use")
arg.add_argument("-l", "--logwandb", help="Log training on wandb", action="store_true")

args = vars(arg.parse_args())

logwandb = args["logwandb"]

############################################
# DEFINE TRAINING FUNCTIONS
############################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

def load_scheduler(scheduler, path):
    scheduler.load_state_dict(torch.load(path))
    return scheduler

def load_optimizer(optimizer, path):
    optimizer.load_state_dict(torch.load(path))
    return optimizer


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

        # PARAMETER DEBUG
        # for param in model.backbone_adapters_MLP_vis[0].up_proj.parameters():
        #     print(param)

        epoch_losses.append(batch_loss.item())

        if logwandb:
            wandb.log({"batch_loss": batch_loss.item()})

    return torch.mean(torch.tensor(epoch_losses)).item()

def train_loop(num_epochs, train_loader, model, criterion, optimizer, scheduler, eval_loader, num_epochs_trained=0):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # create folder for run
    run_path = 'runs/{}'.format(args["name"])
    if not os.path.exists(run_path):
        os.mkdir(run_path)

    best_eval_loss = float('inf')

    loop = tqdm(range(num_epochs), desc="Training locator", leave=True)
    for epoch in loop:
        model.train()

        # TRAIN ONE EPOCH
        epoch_loss = train_one_epoch(epoch, train_loader, model, criterion, optimizer, loop)

        model.eval()

        # EVALUATE MODEL
        eval_losses = []
        with torch.no_grad():
            for samples, bbox in eval_loader:
                images = samples['image'].to(device)
                sentences = clip.tokenize(samples['sentences']).to(device)
                maps, fv = model.encode(images, sentences)

                batch_loss = criterion(maps, bbox['gt'].to(device, dtype=torch.float32))

                eval_losses.append(batch_loss.item())

            eval_loss = torch.mean(torch.tensor(eval_losses)).item()

            loop.write(f'Epoch {epoch+1}/{num_epochs}\tEval loss: {eval_loss:.4f}')

            if logwandb:
                wandb.log({"train_loss": epoch_loss, "eval_loss": eval_loss})

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(model.state_dict(), run_path + "/best.pth")
        
        scheduler.step()

        torch.save(model.state_dict(), run_path + "/epoch_" + str(epoch+num_epochs_trained+1) + ".pth")
        torch.save(optimizer.state_dict(), run_path + "/optimizer_epoch_" + str(epoch+num_epochs_trained+1) + ".pth")
        torch.save(scheduler.state_dict(), run_path + "/scheduler_epoch_" + str(epoch+num_epochs_trained+1) + ".pth")


if __name__ == "__main__":
    ########################################
    # LOAD CLIP MODEL
    ########################################

    model, preprocess = clip.load("ViT-B/16") # only works with ViT-B/16


    ########################################
    # INITIALIZE DATASET
    ########################################

    batch_size = args["batch_size"]
    train_dataset = RefcocogDataset(args["dataset"], split="train", transform=preprocess)
    val_dataset = RefcocogDataset(args["dataset"], split="val", transform=preprocess)
    test_dataset = RefcocogDataset(args["dataset"], split="test", transform=preprocess)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ########################################
    # INITIALIZE TRAINING PARAMETERS
    ########################################

    learning_rate = 5e-5 # 5e-5
    weight_decay = 5e-3 # 5e-3
    num_epochs = args["num_epochs"] # change if epochs alredy trained
    num_epochs_trained = 0 # change if epochs alredy trained


    ########################################
    # INITIALIZE MODEL
    ########################################

    # model.init_adapters() # adds adapters after original state dict has been loaded
    # model.load_parameters(path="") # when needed to resume training
    model.freeze_for_training() # freezes all clip by putting requires_grad=False and then unfreezes adapters

    model = model.to(device)
    model.to(torch.float32)


    ########################################
    # INITIALIZE LOSS FUNCTION, OPTIMIZER AND SCHEDULER
    ########################################

    criterion = FocalDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay, eps=1e-08)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    # optimizer = load_optimizer(optimizer, path="") # when needed to resume training
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=num_epochs)
    # scheduler = load_scheduler(scheduler, path="") # when needed to resume training

    if logwandb:
        wandb.init(project="projectdl", 
                name=args["name"], 
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