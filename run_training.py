import torch
from training import FullTrainer
from data import default_train_transforms, FullLoader
from architecture import FaceNN
from inspect import currentframe, getframeinfo
from pathlib import Path


# performs full training procedure on concatenation of training and validation sets.

# Get paths to train directories. Asssumes folder named data in same directory as this file.

filename = getframeinfo(currentframe()).filename
parent = Path(filename).resolve().parent
train_dir = parent / "data/train"
val_dir = parent / "data/dev"

# Create dataloader
dataloader = FullLoader(train_dir, 
                        val_dir,
                        transform=default_train_transforms,
                        pin_memory=True, 
                        num_workers=8,
                        batch_size=256)

# Create nn
num_classes = dataloader.label_count
facenet = FaceNN(stem_type='classic', num_classes=num_classes).to('cuda')


# Create trainer class.
epochs = 60
output_dir = (parent / "checkpoint")
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "face_nn_fulltrain.pkl"

trainer = FullTrainer(dataloader,
                  facenet, 
                  use_mixup=True, 
                  num_classes=num_classes, 
                  max_epochs=epochs,
                  device='cuda', 
                  save_and_load_filename=output_file)


# Train
if __name__ == "__main__":
    print(f'beginning training for {epochs} epochs. Saving output to {output_file}')
    trainer.train(epochs=epochs)