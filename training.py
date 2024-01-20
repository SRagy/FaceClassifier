from os.path import exists

import torch
from torch import Tensor, device
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer, Adam, AdamW
# from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from torchvision.transforms import v2
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from tqdm import tqdm
from typing import Tuple, Callable
import pickle


class Trainer:
    """Class for training a normalising flow.

    Attributes:
        neural_net (nn.Module) - a neural network to be trained 
        train_losses (list) - a per epoch record of the training loss
        val_losses (list) - a per epoch record of the validation loss.
    """
    def __init__(self,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 neural_net: Module,
                 early_stop_bound: int = 20,
                 max_epochs: int = 300,
                 optimizer: Optimizer = AdamW,
                 base_learning_rate: float = 1e-3,
                 use_lr_scheduler: bool = True,
                 warmup_epochs: int = 0,
                 label_smoothing: float = 0.1,
                 use_cutmix: bool = False,
                 num_classes: int = None,
                 device = torch.device('cpu'),
                 ) -> None:
        """
        Inits Trainer.

        Args:
            train_loader (DataLoader): dataloader class for training data.
            val_loader (DataLoader): dataloader class for validation data.
            neural_net (Module): A CNN or other neural net to be trained.
            early_stop_bound (int, optional): Used in early stopping condition - the number of 
            rounds of no improvement after which to stop. Defaults to 20.
            max_epochs (int, optional): For if early stopping does not occur. Defaults to 300.
            optimizer (Optimizer, optional): Defaults to Adam.
            base_learning_rate (float, optional): Defaults to 3e-4.
            use_lr_scheduler (bool, optional): If True, uses scheduler with cosine decay. Defaults to True.
            warmup_epochs (int, optional): Epochs for linear warmup. Defaults to 20.
            label_smoothing (float, optional): label smoothing for cross entropy loss. Defaults to 0.1.
            use_cutmix (bool, optional): Whether or not to use cutmix&mixup data augmentation. Default None.
            num_classes (int, optional): number of classes. Needed if cutmix is to be used.
            device (Device, optional): cpu or gpu to train on.
        """
        
        self.neural_net = neural_net
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._early_stop_bound = early_stop_bound
        self._max_epochs = max_epochs
        self._trained_epochs = 0
        self._device = device
        self._label_smoothing = label_smoothing
        self._use_cutmix = use_cutmix
        
        # WARNING: currently has num_classes hardcoded. Should edit to read from dataloader.
        if use_cutmix:
            if num_classes is None:
                raise ValueError("num_classes must be defined if using cutmix")
            self._cutmix = v2.CutMix(num_classes=num_classes)
            self._mixup = v2.MixUp(num_classes=num_classes)

        optimisation_parameters = neural_net.parameters()
        self._optimizer = optimizer(optimisation_parameters, lr=base_learning_rate, weight_decay=0.05)
        if use_lr_scheduler:
            self._lr_scheduler=LinearWarmupCosineAnnealingLR(self._optimizer,
                                                             warmup_epochs = warmup_epochs,
                                                             max_epochs = max_epochs,
                                                             warmup_start_lr=1e-3,
                                                             eta_min = 1e-5
                                                            )

        self.train_losses = []
        self.val_errors = []
        self.val_losses = []


    def _loss(self, outputs: Tensor, labels: Tensor, label_smoothing = 0.0):
        """Cross-entropy loss

        Args:
            outputs (Tensor): The output from the neural net classifier
            labels (Tensor): ground truth labels (subject to denoising procedures like mixup)

        Returns:
            Tensor: value of loss.
        """
        return cross_entropy(outputs, labels, label_smoothing = self._label_smoothing)

    def training_loop(self, dataloader: DataLoader):
        """Executes a training epoch.

        Args:
            dataloader (DataLoader): Expects a dataloader with labelled data. 

        Returns:
            Tensor: mean loss
        """
        self.neural_net.train()
        total_loss = 0.
        for images, labels in tqdm(dataloader, desc='Epoch Train Progress'):
            images = images.to(self._device)
            labels = labels.to(self._device)
            
            if self._use_cutmix:
                cutmix_or_mixup = v2.RandomChoice([self._cutmix, self._mixup])
                images, labels = cutmix_or_mixup(images, labels)

            predictions = self.neural_net(images)
            loss = self._loss(predictions, labels, label_smoothing=self._label_smoothing)
            total_loss+=loss.detach()
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
        # mean_loss = total_loss/len(dataloader.dataset) 
        self._lr_scheduler.step()
        return total_loss
    
    
    def validation_loop(self, dataloader: DataLoader):
        self.neural_net.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        loss, correct = 0, 0

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Epoch Val Progress'):
                labels = labels.to(self._device)
                images = images.to(self._device)
                predictions = self.neural_net(images)
                loss += self._loss(predictions, labels)
                correct += (predictions.argmax(1) == labels).to(torch.float).sum()

        loss /= num_batches
        correct /= size
        # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {loss:>4f} \n")
        return loss, correct
    
#     def validation_loop(self, dataloader: DataLoader):
#         """Checks validation loss

#         Args:
#             dataloader (DataLoader): Expects a dataloader with unlabelled data. 
#         """
#         self.neural_net.eval()
#         for params, labels in dataloader:
#             images = images.to(self._device)
#             predictions = neural_net(images)
#             labels = labels.to(self._device)
#             total_loss += self._loss(predictions, labels, label_smoothing=0.0).detach()
#         return total_loss
    
    def log_and_print(self, train_loss, val_loss, val_error, since_improvement):
        self.train_losses.append(train_loss.item())
        self.val_losses.append(val_loss.item())
        self.val_errors.append(val_error.item())
        print(f'\r epoch = {self._trained_epochs}, '
              f'train loss = {train_loss:.3e}, '
              f'percent_correct = {(100 * val_error):>0.1f}%, '
              f'epochs since improvement = {since_improvement}   ', end='')

    def save(self, filename='checkpoint/trainer_state.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
            print(f"saved trainer state at epoch {self._trained_epochs}")

    def load(self, filename='checkpoint/trainer_state.pkl'):
        with open(filename, 'rb') as f:
            self.__dict__ = pickle.load(f)
        print(f"loaded trainer state at epoch {self._trained_epochs}")

    def train(self, epochs = 300):
        """Trains the density estimator. If the total epochs trained 
        already exceeds max_epochs raises an exception.

        Args:
            epochs (int, optional): Max epochs to train for. Defaults to 200.

        Returns:
            Module: Trained normalising flow
        """
        if self._trained_epochs >= self._max_epochs:
            raise Exception(f"Already trained density estimator for the \
                            maximum number of epochs ({self._max_epochs})")

        best_error = torch.inf
        rounds_since_improvement = 0
        for i in range(epochs):
            self._trained_epochs += 1
            train_loss = self.training_loop(self._train_loader)
            val_loss, val_error = self.validation_loop(self._val_loader)
            if val_error < best_error:
                best_error = val_error
                rounds_since_improvement = 0
            else:
                rounds_since_improvement+=1
                

            self.log_and_print(train_loss, val_loss, val_error, rounds_since_improvement)
            self.save()
            if rounds_since_improvement == self._early_stop_bound:
                break
                
            if self._trained_epochs == self._max_epochs:
                break


        # Return for the sake of convenience. Also accessible as attribute of Trainer object.
        return self.neural_net