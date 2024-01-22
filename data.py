import torch
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from PIL import Image
import glob


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
    def __init__(self, directory, transform=v2.ToTensor(), shuffle=True, keep_ratio=1, **kwargs):
        dataset = ImageFolder(directory, transform)
        slice_index = int(keep_ratio*len(dataset))
        self.label_count = len(set(dataset.targets[:slice_index]))
        dataset = Subset(dataset, range(slice_index))
        super().__init__(dataset, shuffle=shuffle, **kwargs)

    @classmethod
    def load_train(cls, transform=v2.ToTensor(), shuffle=True, *args, **kwargs):
        return cls('data/train', transform, shuffle=shuffle, *args, **kwargs)

    @classmethod
    def load_val(cls, *args, **kwargs):
        return cls('data/dev', *args, **kwargs)
    
class FullLoader(DataLoader):
    """
    A dataloader class intended for training with both the train and dev datasets.
    """
    def __init__(self, dir1, dir2, transform=v2.ToTensor(), shuffle=True, keep_ratio=1, **kwargs):
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


def calculate_mean_std(train_loader):
    sum_running_mean = 0
    sum_exp_std = 0
    for num_batches, (img_tensor, _) in enumerate(train_loader):
        sum_running_mean += img_tensor.mean([0,2,3])
        # I choose to calculate E_{images}(std(image)), i.e. expectation of pixel std for each image
        # The alternative would be to calculate the pixel std *across* all images
        sum_exp_std += (img_tensor.std([2,3])).mean(0)

    return sum_running_mean/(num_batches+1), sum_exp_std/(num_batches+1)
