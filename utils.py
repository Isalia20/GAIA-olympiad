import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch.utils.data import Dataset
from matplotlib import patches
from matplotlib.collections import PatchCollection
from metrics import compute_confusion_matrix


######################### არ შეცვალოთ ეს უჯრა ###########################
# უჯრა, რომელიც შეიცავს დამხმარე ფუნქციებს შედეგების ვიზუალიზაციისთვის.

colors = ["red", "green", "blue", "yellow", "black", "purple", "orange", "brown", "pink"]
label_names = ['1 grosz', '2 grosze', '5 groszy', '10 groszy', '20 groszy', '50 groszy', '1 złotych', '2 złote', '5 złotych']

def show_sample(
        sample: dict
    ):
    """
    ფუნქცია, რომელიც აჩვენებს სურათს ობიექტების bounding box-ებით.

    იღებს:
        sample: ლექსიკონი, რომელიც შეიცავს სურათს და იარლიყებს.
    """
    image = sample['image']
    meta = sample

    plt.figure(figsize=(9, 5))
    plt.imshow(image.permute((1, 2, 0)))
    plt.xticks([])
    plt.yticks([])

    if meta is not None:
        patches_list = []
        legend_labels = []

        for bbox, label in zip(meta['boxes'], meta['labels']):
            points = np.array(bbox)
            points = points.astype(int)

            # მართკუთხედის დახატვა, რომელიც ობიექტს აკრავს
            rect = patches.Rectangle(
                (points[0], points[1]),
                points[2] - points[0],
                points[3] - points[1],
                linewidth=2,
                edgecolor=colors[label.item()],
                facecolor='none'
            )
            patches_list.append(rect)

            # ლეგენდა უნიკალური იარლიყებით
            if label_names[label.item()] not in legend_labels:
                legend_labels.append(label_names[label.item()])

        patch_collection = PatchCollection(patches_list, match_original=True)
        plt.gca().add_collection(patch_collection)

        # ლეგენდის დამატება უნიკალური იარლიყებით
        handles = [patches.Patch(color=colors[i], label=label) for i, label in enumerate(label_names) if label in legend_labels]
        plt.legend(handles=handles, loc="upper right")

    plt.show()

def plot_detection_results_grid(
        predictions: list,
        ds: Dataset,
        size: tuple = (2, 3)
    ):
    """
    ფუნქცია, რომელიც აჩვენებს დეტექციის შედეგებს არჩეულ სავალიდაციო სურათებზე.

    იღებს:
        predictions: სია, რომელიც შეიცავს პროგნოზირებულ bounding box-ებს მონაცემთა ნაკრების თითოეული ნიმუშისთვის.
        ds: მონაცემთა ნაკრები.
        size (tuple, არასავალდებულო): სურათების ბადის ზომა.
    """

    fig, axes = plt.subplots(size[0], size[1], figsize=(size[1]*6, size[0]*5))

    for i, (sample, pred_meta) in enumerate(zip(ds, predictions)):
        if i >= size[0] * size[1]:
            break

        img = sample["image"]

        ax = axes[i // size[1], i % size[1]]
        ax.imshow(img.permute(1, 2, 0))
        ax.axis('off')

        for x1, y1, x2, y2, label, _ in pred_meta:
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=colors[label],
                facecolor='none',
            )
            ax.add_patch(rect)

    handles = [patches.Patch(color=colors[i], label=label) for i, label in enumerate(label_names)]
    plt.legend(handles=handles, loc='upper right')
    plt.suptitle("მონეტების დეტექცია არჩეულ ექვს სავალიდაციო სურათზე")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(
        predictions: list,
        ds: Dataset,
        iou_threshold: float = 0.5
    ):
    """
    ფუნქცია, რომელიც აჩვენებს შეცდომათა მატრიცას მონეტების დეტექციისთვის.

    იღებს:
        predictions: მონეტების დეტექციის შედეგი.
        ds: მონაცემთა ნაკრები.
        iou_threshold float: IoU-ს ზღურბლი პროგნოზის ობიექტთან მინიჭებისთვის.
    """
    conf_matrix = compute_confusion_matrix(predictions, ds, iou_threshold=iou_threshold)

    labels = ['1 grosz', '2 grosze', '5 groszy', '10 groszy', '20 groszy', '50 groszy', '1 złotych', '2 złote', '5 złotych',"brak (tło)"]
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)

    plt.xlabel("მოდელმა იპროგნოზირა", labelpad=15)
    plt.ylabel("მაშინ როცა უნდა ეპროგნოზირებინა", labelpad=15)
    plt.xticks(rotation=45)

    plt.title("შეცდომათა მატრიცა მონეტების დეტექციისთვის IOU={}-ის გამოყენებით".format(iou_threshold))
    plt.show()