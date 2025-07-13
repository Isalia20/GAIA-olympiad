######################### არ შეცვალოთ ეს უჯრა ##########################

import os
import torch
import pickle
import gdown
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms.v2 as T
from collections.abc import Callable
from matplotlib import patches
from matplotlib.collections import PatchCollection
from torchvision.models import resnet18
from torchvision.ops import box_iou
from torch.utils.data import Dataset
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from dataset import setup_data
from utils import show_sample
from metrics import predict_all_bounding_boxes, calculate_map
from utils import plot_confusion_matrix, plot_detection_results_grid
from torchvision.models.detection import fcos_resnet50_fpn
from metrics import DEVICE

# assert torch.cuda.is_available(), "CUDA niedostępna!"

######################### არ შეცვალოთ ეს უჯრა ##########################

seed = 12345

os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# tაქ შეგიძლიათ განახორციელოთ თქვენი დეტექტორი, რომელიც იპოვის ობიექტების სიას ფორმატში (x1, y1, x2, y2, label, confidence)

class YourDetector(nn.Module):
    def __init__(self):
        # აქ შეგიძლიათ თქვენი მოდელის ინიციალიზაცია
        super(YourDetector, self).__init__()
        self.detector_model = fcos_resnet50_fpn(weights=None, weights_backbone=None, trainable_backbone_layers=5, num_classes=9)

    def forward(
            self,
            img: torch.Tensor,
            target = None,
        ) -> list:
        """
        თქვენი მონეტების დეტექციის ფუნქცია გამოსახულებაზე.
        იღებს:
            img: დასამუშავებელი გამოსახულება.
        აბრუნებს:
            ნაპოვნი ობიექტების სიას კორტეტის სახით (x1, y1, x2, y2, label, confidence).
        """
        if self.training:
            out = self.detector_model(img, target)
            return out
        else:
            out = self.detector_model([img])[0]
            boxes = out["boxes"]  # n_pred_objects, 4
            scores = out["scores"]  # n_pred_scores
            labels = out["labels"]  # n_pred_labels
            return [(box[0].item(), box[1].item(), box[2].item(), box[3].item(), label.item(), score.item()) for box, score, label in zip(boxes, scores, labels)]

def train_detector(
        model,
        train_ds: Dataset
    ):
    # მოვამზადოთ მოდელი, ოპტიმიზატორი და დანაკარგის ფუნქცია (მოძრავი ფანჯრის ტექნიკის გამოყენებით ჩვენი მოდელის სწავლების პრობლემა კლასიფიკაციის პრობლემად იქცა)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # მოვამზადოთ dataloader-ები
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=lambda x: x)

    # ვავარჯიშოთ მოდელი არჩეული ეპოქების რაოდენობის განმავლობაში
    epochs = 1

    pbar = tqdm(range(epochs), desc="ვარჯიში", total=epochs)
    for _ in pbar:
        epoch_losses = []

        for batch in train_dl:
            images = torch.stack([i["image"] for i in batch]).to(DEVICE)
            targets = [{"boxes": i["boxes"].to(DEVICE), "labels": i["labels"].to(DEVICE)} for i in batch]

            optimizer.zero_grad()
            output = model(images, targets)
            loss = output["classification"] + output["bbox_regression"] + output["bbox_ctrness"]
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.detach().cpu().item())

        avg_loss = np.mean(epoch_losses)
        pbar.set_postfix({'train loss': avg_loss})

    model.eval()
    return model


def main():
    train_ds, val_ds = setup_data(root='./data/')
    model = YourDetector()
    train_detector(model, train_ds)
    out = predict_all_bounding_boxes(model, val_ds)
    map_val = calculate_map(out, val_ds, return_all=True)
    print(f"mAP IoU=0.5 ზღვრის გამოყენებით ვალიდაციის ნაკრებზე: {map_val['map_50'].item():.2f}")
    print(f"mAP IoU-ს მრავალი მნიშვნელობისთვის ვალიდაციის ნაკრებზე:  {map_val['map']:.2f}, ეს არის მეტრიკა, რომელიც ფასდება კონკურსში. თქვენი ამოცანაა მისი მაქსიმიზაცია.")


if __name__ == "__main__":
    main()
