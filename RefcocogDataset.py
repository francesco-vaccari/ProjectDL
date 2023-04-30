import json
import clip
import pandas
from torch.utils.data import random_split
from typing import Sequence, Union

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class RefcocogDataset(Dataset):
    def __init__(self, base_path, split=None, transform=None):
        annotation_path = base_path + "/annotations/"

        self.IMAGES_PATH = base_path + "/images/"
        self.transform = transform

        tmp_annotations = pandas.read_pickle(annotation_path + "refs(umd).p")
        tmp_instances = json.load(open(annotation_path + "instances.json", "r"))

        annotations_dt = pandas.DataFrame.from_records(tmp_annotations) \
            .filter(items=["image_id", "split", "sentences", "ann_id"])

        instances_dt = pandas.DataFrame.from_records(tmp_instances['annotations'])

        self.annotations = annotations_dt \
            .merge(instances_dt[["id", "bbox", "area"]], left_on="ann_id", right_on="id") \
            .drop(columns="id")

        if split is not None:
            if split.lower() == 'train':
                self.annotations = self.__get_train_annotations()

            if split.lower() == 'test':
                self.annotations = self.__get_test_annotations()

    def splitTrainVal(self, lengths: Sequence[Union[int, float]]):
        return random_split(self, lengths)

    def getImage(self, idx):
        id = idx[0].item()
        item = self.annotations.iloc[id]
        image = self.__getimage(item.image_id)

        return image

    def __get_train_annotations(self):
        return self.annotations[self.annotations.split == "train"].reset_index()

    def __get_test_annotations(self):
        return self.annotations[self.annotations.split == "test"].reset_index()

    def __getimage(self, id):
        return Image.open(self.IMAGES_PATH + "COCO_train2014_" + str(id).zfill(12) + ".jpg")

    def __extract_sentences(self, sentences):
        return [f"a photo of a {s['sent']}" for s in sentences]

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, idx):
        item = self.annotations.iloc[idx]
        image = self.__getimage(item.image_id)
        sentences = self.__extract_sentences(item.sentences)

        if self.transform:
            image = self.transform(image)

        sample = {'idx': idx, 'image': image, 'sentences': sentences}

        return sample, item.bbox


if __name__ == "__main__":
    _, preprocess = clip.load("ViT-B/32")

    dataset = RefcocogDataset("../refcocog", split="train", transform=preprocess)
    train, val = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train)

    train_features, train_bbox = next(iter(train_dataloader))
    dataset.getImage(train_features['idx']).show()
    print(len(train))
    print(len(val))
