import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
from dataset.RefcocogDataset import RefcocogDataset
from model.refiner.refiner import Refiner


locator_path = "./models/locator_epoch_6.pth"
refiner_path = "./models/refiner_epoch_1.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

locator, preprocess = clip.load("ViT-B/16")
locator.init_adapters()
locator.load_state_dict(torch.load(locator_path, map_location=device))
locator = locator.to(device)

refiner = Refiner()
refiner.load_state_dict(torch.load(refiner_path, map_location=device))
refiner = refiner.to(device)

test_dataset = RefcocogDataset("./dataset/refcocog", split="test", transform=preprocess)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


def extract_bbox(out):
    map = out.squeeze(0).squeeze(0).detach().cpu().numpy()
    # normalize map to [0, 1]
    map = (map - map.min()) / (map.max() - map.min())
    # threshold map
    map = (map > 0.8)
    x_min = 225
    y_min = 225
    x_max = 0
    y_max = 0
    for i in range(224):
        for j in range(224):
            if map[i][j] == True:
                if i < y_min: y_min = i
                if i > y_max: y_max = i
                if j < x_min: x_min = j
                if j > x_max: x_max = j
    
    return x_min, y_min, x_max, y_max
                

def computeIntersection(fx1, fy1, fx2, fy2, sx1, sy1, sx2, sy2):
    dx = min(fx2, sx2) - max(fx1, sx1)
    dy = min(fy2, sy2) - max(fy1, sy1)
    if (dx>=0) and (dy>=0):
        area = dx*dy
    else:
        area = 0
    return area

def compute_accuracy(out, bbox):
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    x_min, y_min, x_max, y_max = extract_bbox(out)
    intersection = computeIntersection(x_min, y_min, x_max, y_max, x, y, x+w, y+h)
    area1 = (x_max-x_min)*(y_max-y_min)
    area2 = w*h
    union = area1 + area2 - intersection
    return intersection / union


acc = []
for sample, bbox in test_loader:
    image = sample['image'].to(device)
    sentences = clip.tokenize(sample['sentences']).to(device)
    maps, fv = locator.encode(image, sentences)
    
    out = refiner(maps, fv)

    for idx in range(out.shape[0]):
        box = bbox['bbox'][0][idx].item(), bbox['bbox'][1][idx].item(), bbox['bbox'][2][idx].item(), bbox['bbox'][3][idx].item()
        accuracy = compute_accuracy(out[idx], box)
        acc.append(accuracy)

print(f'Accuracy : {sum(acc)/len(acc)}')