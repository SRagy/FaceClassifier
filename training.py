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
    """Class for supervised training of a neural network classifier.

    Attributes:
        neural_net (nn.Module) - a neural network to be trained 
        train_losses (list) - a per epoch record of the training loss
        val_losses (list) - a per epoch record of the validation loss.
        correct_fracs (list) - a per epoch record of correctness rate.
    """
    def __init__(self,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 neural_net: Module,
                 early_stop_bound: int = 20,
                 max_epochs: int = 60,
                 optimizer: Optimizer = AdamW,
                 base_learning_rate: float = 1e-3,
                 use_lr_scheduler: bool = True,
                 use_amp: bool = True,
                 warmup_epochs: int = 10,
                 label_smoothing: float = 0.1,
                 use_mixup: bool = False,
                 num_classes: int = None,
                 device: device = torch.device('cpu'),
                 save_and_load_filename: str = 'checkpoint/trainer_state.pkl'
                 ) -> None:
        """
        Inits Trainer.

        Args:
            train_loader (DataLoader): dataloader class for training data.
            val_loader (DataLoader): dataloader class for validation data.
            neural_net (Module): A CNN or other neural net to be trained.
            early_stop_bound (int, optional): Used in early stopping condition - the number of 
            rounds of no improvement after which to stop. Defaults to 20.
            max_epochs (int, optional): For if early stopping does not occur. Defaults to 60.
            optimizer (Optimizer, optional): Defaults to AdamW.
            base_learning_rate (float, optional): Defaults to 3e-4.
            use_lr_scheduler (bool, optional): If True, uses scheduler with cosine decay. Defaults to True.
            use_amp (bool, optional): Whether to use automatic mixed precision. Defaults to True.
            warmup_epochs (int, optional): Epochs for linear warmup. Defaults to 20.
            label_smoothing (float, optional): label smoothing for cross entropy loss. Defaults to 0.1.
            use_mixup (bool, optional): Whether or not to use mixup data augmentation. Default None.
            num_classes (int, optional): number of classes. Needed if mixup is to be used.
            device (Device, optional): cpu or gpu to train on.
            save_and_load_filename (str, optional): directory for saving class instances.
        """
        
        self.neural_net = neural_net
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._early_stop_bound = early_stop_bound
        self._max_epochs = max_epochs
        self._trained_epochs = 0
        self._device = device
        self._label_smoothing = label_smoothing
        self._use_lr_scheduler = use_lr_scheduler
        self._use_mixup = use_mixup
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self._autocaster = torch.autocast(device_type=self._device, dtype=torch.float16, enabled=use_amp)
        self._save_and_load_filename = save_and_load_filename
        
        if use_mixup:
            if num_classes is None:
                raise ValueError("num_classes must be defined if using cutmix")
            # self._cutmix = v2.CutMix(num_classes=num_classes)
            self._mixup = v2.MixUp(num_classes=num_classes)

        optimisation_parameters = neural_net.parameters()
        self._optimizer = optimizer(optimisation_parameters, lr=base_learning_rate, weight_decay=0.05)
        if use_lr_scheduler:
            self._lr_scheduler=LinearWarmupCosineAnnealingLR(self._optimizer,
                                                             warmup_epochs = warmup_epochs,
                                                             max_epochs = max_epochs,
                                                             warmup_start_lr=3e-4,
                                                             eta_min = 1e-5
                                                            )

        self.train_losses = []
        self.correct_fracs = []
        self.val_losses = []


    def _loss(self, outputs: Tensor, labels: Tensor, label_smoothing: float = 0.0):
        """Cross-entropy loss

        Args:
            outputs (Tensor): The output from the neural net classifier
            labels (Tensor): ground truth labels (subject to denoising procedures like mixup)
            label_smoothing (float): label smoothing parameter for cross entropy loss.

        Returns:
            Tensor: value of loss.
        """
        return cross_entropy(outputs, labels, label_smoothing = label_smoothing)

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
            with self._autocaster:
                images = images.to(self._device)
                labels = labels.to(self._device)

                if self._use_mixup: 
                    # cutmix_or_mixup = v2.RandomChoice([self._cutmix, self._mixup])
                    images, labels = self._mixup(images, labels)

                predictions = self.neural_net(images)
                loss = self._loss(predictions, labels, label_smoothing=self._label_smoothing)
            total_loss+=loss.detach()
            self._optimizer.zero_grad()
            self._scaler.scale(loss).backward()
            self._scaler.step(self._optimizer)
            self._scaler.update()
        # mean_loss = total_loss/len(dataloader.dataset) 
        if self._use_lr_scheduler:
            self._lr_scheduler.step()
        # should probably count samples in loop explicitly, but this works for now.
        mean_loss = total_loss/len(dataloader.dataset) 
        return mean_loss
    
    
    def validation_loop(self, dataloader: DataLoader):
        """Calculates the loss function on the validation set, as well as the percentage of
        images correctly classfied

        Args:
            dataloader (DataLoader): A dataloader with labelled data.

        Returns:
            Tuple(Tensor, Tensor): (validation loss, classication success ratio)
        """
        self.neural_net.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        loss, correct = 0, 0
    
        with self._autocaster:
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
        
    def log_and_print(self, train_loss: Tensor,
                      val_loss: Tensor, 
                      correct_frac: Tensor, 
                      since_improvement: int):
        """Record loss vlaues and print for current epoch.

        Args:
            train_loss (Tensor): The training loss for the current epoch
            val_loss (Tensor): The validation loss for the current epoch
            correct_frac (Tensor): The classification success ratio
            since_improvement (int): Number of rounds val_loss last decreased
        """
        self.train_losses.append(train_loss.item())
        self.val_losses.append(val_loss.item())
        self.correct_fracs.append(correct_frac.item())
        if self._use_lr_scheduler:
            lr = self._lr_scheduler.get_last_lr()[0]
        else:
            lr = 'base'
        print(f'epoch = {self._trained_epochs}, '
              f'train loss = {train_loss:.3e}, '
              f'percent_correct = {(100 * correct_frac):>0.1f}%, '
              f'epochs since improvement = {since_improvement}, '
              f'lr={lr:.3e}', end='\n')

    def save(self):
        """Saves the trainer state to self._save_and_load_filename
        """
        filename = self._save_and_load_filename
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
            print(f"saved trainer state at epoch {self._trained_epochs}")

    def load(self):
        """Loads the trainer state from self._save_and_load_filename
        """
        filename = self._save_and_load_filename
        with open(filename, 'rb') as f:
            self.__dict__ = pickle.load(f)
        print(f"loaded trainer state at epoch {self._trained_epochs}")

    def train(self, epochs = 60):
        """Trains the neural net. If the total epochs trained 
        already exceeds max_epochs raises an exception.

        Args:
            epochs (int, optional): Max epochs to train for. Defaults to 60.

        Returns:
            Module: Trained neural net
        """
        if self._trained_epochs >= self._max_epochs:
            raise Exception(f"Already trained density estimator for the \
                            maximum number of epochs ({self._max_epochs})")

        best_loss = torch.inf
        rounds_since_improvement = 0
        for i in range(epochs):
            self._trained_epochs += 1
            train_loss = self.training_loop(self._train_loader)
            val_loss, correct_frac = self.validation_loop(self._val_loader)
            if val_loss < best_loss:
                best_loss = val_loss
                rounds_since_improvement = 0
            else:
                rounds_since_improvement+=1
                

            self.log_and_print(train_loss, val_loss, correct_frac, rounds_since_improvement)
            self.save()
            if rounds_since_improvement == self._early_stop_bound:
                break
                
            if self._trained_epochs == self._max_epochs:
                break


        # Return for the sake of convenience. Also accessible as attribute of Trainer object.
        return self.neural_net
    
class FullTrainer(Trainer):
    """Class for training a supervised neural network on the train+dev sets together

    Attributes:
        neural_net (nn.Module) - a neural network to be trained 
        train_losses (list) - a per epoch record of the training loss
    """
    def __init__(self,
                 train_loader: DataLoader,
                 neural_net: Module,
                 early_stop_bound: int = 20,
                 max_epochs: int = 50,
                 optimizer: Optimizer = AdamW,
                 base_learning_rate: float = 1e-3,
                 use_lr_scheduler: bool = True,
                 use_amp: bool = True,
                 warmup_epochs: int = 10,
                 label_smoothing: float = 0.1,
                 use_mixup: bool = False,
                 num_classes: int = None,
                 device = torch.device('cpu'),
                 save_and_load_filename: str = 'checkpoint/trainer_state.pkl'
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
            optimizer (Optimizer, optional): Defaults to AdamW.
            base_learning_rate (float, optional): Defaults to 3e-4.
            use_lr_scheduler (bool, optional): If True, uses scheduler with cosine decay. Defaults to True.
            use_amp (bool, optional): Whether to use automatic mixed precision. Defaults to True.
            warmup_epochs (int, optional): Epochs for linear warmup. Defaults to 20.
            label_smoothing (float, optional): label smoothing for cross entropy loss. Defaults to 0.1.
            use_mixup (bool, optional): Whether or not to use mixup data augmentation. Default None.
            num_classes (int, optional): number of classes. Needed if mixup is to be used.
            device (Device, optional): cpu or gpu to train on.
            save_and_load_filename (str, optional): directory for saving class instances.
        """
        super().__init__(train_loader,
                       None,
                       neural_net,
                       early_stop_bound,
                       max_epochs,
                       optimizer,
                       base_learning_rate,
                       use_lr_scheduler,
                       use_amp,
                       warmup_epochs,
                       label_smoothing,
                       use_mixup,
                       num_classes,
                       device,
                       save_and_load_filename)
        
    def log_and_print(self, train_loss):
        self.train_losses.append(train_loss.item())
        if self._use_lr_scheduler:
            lr = self._lr_scheduler.get_last_lr()[0]
        else:
            lr = 'base'
        print(f'epoch = {self._trained_epochs}, '
              f'train loss = {train_loss:.3e}, '
              f'lr={lr:.3e}', end='\n')


    def train(self, epochs = None):
        """Trains the density estimator. If the total epochs trained 
        already exceeds max_epochs raises an exception.

        Args:
            epochs (int, optional): Max epochs to train for. Defaults to None.

        Returns:
            Module: Trained normalising neural net
        """
        if epochs is None:
            epochs = self._max_epochs
        
        if self._trained_epochs >= self._max_epochs:
            raise Exception(f"Already trained density estimator for the \
                            maximum number of epochs ({self._max_epochs})")

        for _ in range(epochs):
            self._trained_epochs += 1
            train_loss = self.training_loop(self._train_loader)
            self.log_and_print(train_loss)
            self.save()
            if self._trained_epochs == self._max_epochs:
                break
