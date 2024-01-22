import torch
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from typing import Callable
from PIL import Image
import glob

# Default transformations for training and validation sets.
default_train_transforms = v2.Compose([v2.RandomRotation(30),
                                       v2.RandomHorizontalFlip(),
                                       v2.ColorJitter(0.2,0.2,0.2,0.2),
                                       v2.ToImage(), 
                                       v2.ToDtype(torch.float32, scale=True),
                                       v2.Normalize([0.5103, 0.4014, 0.3509], [0.2708, 0.2363, 0.2226])])

default_val_transforms  = v2.Compose([v2.ToImage(),
                                      v2.ToDtype(torch.float32, scale=True), 
                                      v2.Normalize([0.5103, 0.4014, 0.3509], [0.2708, 0.2363, 0.2226])])


class DefaultLoader(DataLoader):
    """dataloader getter for ImageFolder type datasets.
    """
    def __init__(self, directory: str, 
                 transform: Callable = v2.ToTensor(), 
                 shuffle: bool = True, 
                 keep_ratio: float = 1, 
                 **kwargs):
        """A class for getting dataloaders for ImageFolder datasets.

        Args:
            directory (str): the location of the image folder
            transform (Callable, optional): The transform to apply to the PIL images. Defaults to v2.ToTensor().
            shuffle (bool, optional): dataloader shuffle parameter. Defaults to True.
            keep_ratio (float, optional): Amount of the dataset to keep. Useful for prototyping. Defaults to 1.
        """
        dataset = ImageFolder(directory, transform)
        slice_index = int(keep_ratio*len(dataset))
        self.label_count = len(set(dataset.targets[:slice_index]))
        dataset = Subset(dataset, range(slice_index))
        super().__init__(dataset, shuffle=shuffle, **kwargs)

    @classmethod
    def load_train(cls, transform=v2.ToTensor(), shuffle=True, *args, **kwargs):
        """Factory method for loading train set from default directory.

        Args:
            transform (Callable, optional): The transform to apply to the PIL images. Defaults to v2.ToTensor().
            shuffle (bool, optional): dataloader shuffle parameter. Defaults to True.

        Returns:
            DataLoader: dataloader instance.
        """
        return cls('data/train', transform, shuffle=shuffle, *args, **kwargs)

    @classmethod
    def load_val(cls, *args, **kwargs):
        """Factory method for loading validation set from default directory.

        Returns:
            DataLoader: dataloader instance.
        """
        return cls('data/dev', *args, **kwargs)
    
class FullLoader(DataLoader):
    """
    A dataloader class intended for training with both the train and dev datasets.
    """
    def __init__(self, dir1: str, 
                 dir2: str, 
                 transform: Callable = v2.ToTensor(), 
                 shuffle: bool = True, 
                 keep_ratio: int = 1, 
                 **kwargs):
        """_summary_

        Args:
            dir1 (str): _description_
            dir2 (str): _description_
            transform (Callable, optional): _description_. Defaults to v2.ToTensor().
            shuffle (bool, optional): _description_. Defaults to True.
            keep_ratio (int, optional): _description_. Defaults to 1.
        """
        dataset1 = ImageFolder(dir1, transform)
        dataset2 = ImageFolder(dir2, transform)
        dataset = ConcatDataset([dataset1, dataset2])
        slice_index = int(keep_ratio*len(dataset))
        self.label_count = len(set(dataset1.targets[:slice_index])) # only works if both datasets the same
        dataset = Subset(dataset, range(slice_index))
        super().__init__(dataset, shuffle=shuffle, **kwargs)


class ImageLoader(Dataset):
    """A simple dataset for loading unlabelled images, as in
    
    https://discuss.pytorch.org/t/load-my-own-test-data-images/30009/2
    """
    def __init__(self, directory='data/test', transform=v2.ToTensor()):
        self.image_paths = glob.glob(directory + '*.jpg')
        self.transform = transform

    def __getitem__(self, index):
        x = Image.open(self.image_paths[index])
        if self.transform is not None:
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.image_paths)
    

class DefaultTest(DataLoader):
    def __init__(self, directory='data/test'):
        dataset = ImageLoader(directory)
        super().__init__(dataset)


def calculate_mean_std(train_loader: DataLoader):
    """Calculate mean and deviation for normalizing images.

    Args:
        train_loader (Dataloader): The dataloader for the image dataset.

    Returns:
        Tuple: (mean, std)
    """
    sum_running_mean = 0
    sum_exp_std = 0
    for num_batches, (img_tensor, _) in enumerate(train_loader):
        sum_running_mean += img_tensor.mean([0,2,3])
        # I choose to calculate E_{images}(std(image)), i.e. expectation of pixel std for each image
        # The alternative would be to calculate the pixel std *across* all images
        sum_exp_std += (img_tensor.std([2,3])).mean(0)

    return sum_running_mean/(num_batches+1), sum_exp_std/(num_batches+1)
