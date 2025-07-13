######################### არ შეცვალოთ ეს უჯრა ###########################
# უჯრა, რომელიც შეიცავს დამხმარე ფუნქციებს მონაცემების მოსამზადებლად.
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset
from typing import Callable
import pickle
import os
import gdown


class CoinsDataset(Dataset):
    """
    მონეტების მონაცემთა ნაკრები, რომელიც იტვირთება pickle ფაილიდან.

    იღებს:
        pickle_file (str): pickle ფაილის მისამართი, რომელიც შეიცავს მონაცემებს.
        transform (callable, არასავალდებულო): ტრანსფორმაციები, რომლებიც გამოიყენება სურათებსა და იარლიყებზე.
    """
    def __init__(
            self,
            pickle_file: str,
            transform: Callable | None = None
        ):
        self.transform = transform

        with open(pickle_file, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self) -> int:
        """აბრუნებს ნიმუშების რაოდენობას მონაცემთა ნაკრებში."""
        return len(self.data)

    def __getitem__(
            self,
            idx: int
        ) -> dict:
        """
        იღებს მონაცემთა ნიმუშს ინდექსის მიხედვით.

        იღებს:
            idx (int): ნიმუშის ინდექსი.

        აბრუნებს:
            dict: ლექსიკონი, რომელიც შეიცავს სურათს და მის შესაბამის სამიზნე ობიექტებს (boxes, labels).
        """
        sample = self.data[idx]
        image = sample['image']
        target = {
            'boxes': sample['boxes'],
            'labels': sample['labels']
        }

        if self.transform:
            image, target = self.transform(image, target)

        return {
            'image': image,
            **target
        }

def setup_data(
        train_transform: Callable | None = None,
        val_transform: Callable | None = None,
        root: str = 'data/'
    ) -> tuple:
    """
    ამზადებს მონაცემთა ნაკრებებს სავარჯიშოდ და ვალიდაციისთვის, საჭიროების შემთხვევაში იწერს მათ.

    იღებს:
        train_transform (callable, არასავალდებულო): აუგმენტაციები სავარჯიშო ნაკრებისთვის.
        val_transform (callable, არასავალდებულო): აუგმენტაციები სავალიდაციო ნაკრებისთვის.
        root (str, არასავალდებულო): საბაზისო კატალოგი მონაცემთა ფაილებისთვის.

    აბრუნებს:
        tuple: მონაცემთა ნაკრებები (train_ds, val_ds).
    """
    if train_transform is None:
        train_transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    if val_transform is None:
        val_transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

    train_file = root+'train.pkl'
    val_file = root+'val.pkl'

    if not os.path.exists(root):
        os.makedirs(root)

    if not os.path.exists(train_file):
        url = "https://drive.google.com/uc?id=1KC8FBlCuwh9ITUt0CcPeRJCpqy5j4WBp"
        gdown.download(url, train_file, quiet=True)

    if not os.path.exists(val_file):
        url = "https://drive.google.com/uc?id=1Oza4UjnmAUeae2cA8YDwMWxVHOb7SKdP"
        gdown.download(url, val_file, quiet=True)

    train_ds = CoinsDataset(root+'train.pkl', transform=train_transform)
    val_ds = CoinsDataset(root+'val.pkl', transform=val_transform)

    return train_ds, val_ds
