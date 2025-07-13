import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from torchvision.ops import box_iou
from torchmetrics.detection import MeanAveragePrecision


DEVICE = "mps"

def predict_all_bounding_boxes(
        model: nn.Module,
        ds: Dataset
    ) -> list:
    """
    ფუნქცია, რომელიც პროგნოზირებს ყველა bounding box-ს მონაცემთა ნაკრებისთვის მოდელის გამოყენებით.

    იღებს:
        model: ობიექტების დეტექციის მოდელი.
        ds: მონაცემთა ნაკრები.

    აბრუნებს:
        სია, რომელიც შეიცავს ყველა პროგნოზირებულ bounding box-ს მონაცემთა ნაკრების თითოეული ნიმუშისთვის.
    """
    all_pred_bboxes = []

    for sample in ds:
        img = sample["image"].to(DEVICE)
        pred_bboxes = model(img)
        all_pred_bboxes.append(pred_bboxes)

    return all_pred_bboxes

def calculate_map(
        predictions: list,
        ds: Dataset,
        return_all: bool = False
    ) -> dict | float:
    """
    ფუნქცია, რომელიც ითვლის საშუალო სიზუსტეს (mAP) პროგნოზირებული bounding box-ებისთვის მთელი მონაცემთა ნაკრებისთვის.

    იღებს:
        predictions: სია, რომელიც შეიცავს პროგნოზირებულ bounding box-ებს მონაცემთა ნაკრების თითოეული ნიმუშისთვის.
        ds: მონაცემთა ნაკრები, რომელიც შეიცავს რეალურ bounding box-ებს.
        return_all (bool, არასავალდებულო): დააბრუნოს თუ არა ყველა მეტრიკა, თუ მხოლოდ mAP:0.5:0.95:0.05.

    აბრუნებს:
        mAP-ის მნიშვნელობა ან ყველა მეტრიკა ლექსიკონის სახით.
    """
    meta = []

    for img_meta in predictions:
        entry = {
            "boxes": [],
            "labels": [],
            "scores": []
        }

        for box in img_meta:
            entry["boxes"].append(box[:4])
            entry["labels"].append(box[4])
            entry["scores"].append(box[5])

        meta.append(entry)

    for i in range(len(meta)):
        meta[i]['boxes'] = torch.tensor(meta[i]['boxes'])
        meta[i]['labels'] = torch.tensor(meta[i]['labels']).view(-1)
        meta[i]['scores'] = torch.tensor(meta[i]['scores'])

    mAP = MeanAveragePrecision()

    GT = [{
        "boxes": sample["boxes"],
        "labels": sample["labels"]
    } for sample in ds]

    output = mAP(meta, GT)

    if return_all:
        return output

    return mAP(meta, GT)['map'].item()

def compute_confusion_matrix(
        predictions: list,
        ds: Dataset,
        iou_threshold: float = 0.5
    ) -> np.ndarray:
    """
    ფუნქცია, რომელიც ითვლის შეცდომათა მატრიცას პროგნოზირებული bounding box-ებისთვის მთელი მონაცემთა ნაკრებისთვის.

    იღებს:
        predictions: სია, რომელიც შეიცავს პროგნოზირებულ bounding box-ებს მონაცემთა ნაკრების თითოეული ნიმუშისთვის.
        ds: მონაცემთა ნაკრები, რომელიც შეიცავს რეალურ bounding box-ებს.
        iou_threshold (float, არასავალდებულო): IoU-ს ზღურბლი პროგნოზის ობიექტთან მინიჭებისთვის.

    აბრუნებს:
        შეცდომათა მატრიცა.
    """
    num_classes = 10  # 9 მონეტის კლასი + 1 ფონი
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for pred_boxes, item in zip(predictions, ds):
        # Ground truth (რეალური მონაცემები)
        gt_boxes = item['boxes']
        gt_labels = item['labels']

        # პროგნოზები
        pred_boxes_tensor = torch.tensor([p[:4] for p in pred_boxes]) if pred_boxes else torch.empty((0, 4))
        pred_labels = torch.tensor([p[4] for p in pred_boxes]) if pred_boxes else torch.empty((0,), dtype=torch.long)

        # IoU მნიშვნელობები პროგნოზებისა და ground truth-ის ყველა წყვილისთვის
        iou_matrix = box_iou(pred_boxes_tensor, gt_boxes) if pred_boxes else torch.empty((0, gt_boxes.shape[0]))

        # IoU-ს საფუძველზე მივანიჭოთ პროგნოზები ground truth-ს
        matched_gt = set()
        for pred_idx, ious in enumerate(iou_matrix):
            max_iou, gt_idx = torch.max(ious, dim=0)
            if max_iou >= iou_threshold and gt_idx.item() not in matched_gt:
                conf_matrix[gt_labels[gt_idx].item(), pred_labels[pred_idx].item()] += 1
                matched_gt.add(gt_idx.item())
            else:
                conf_matrix[-1, pred_labels[pred_idx].item()] += 1  # False positive

        # ყველა მიუნიჭებელი ground truth ობიექტისთვის დავამატოთ როგორც False negative
        for gt_idx in range(len(gt_boxes)):
            if gt_idx not in matched_gt:
                conf_matrix[gt_labels[gt_idx].item(), -1] += 1

    return conf_matrix
