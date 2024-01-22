# A small face classifier with a modern architecture
This project produces a modern architecture for face classification, with a view to keeping it small (it comes to less than 20m parameters), and easily trainable on a 16GB GPU.

The architecture is heavily inspired by, but not identical to, ConvNeXt, which is a fairly recent CNN-based network design, itself inspired by vision transformers with which it has comparable performance. I have found my architecture appears to perform slightly better than the base ConvNeXt design for my training procedure, but I haven't investigated this in sufficient depth to make the claim with full confidence.

Trained weights are available on request.

### Highlights
The project leverages recent design and training techniques. It uses the block design from the ConvNeXt architecture, with some small variations for type and position of norm layers. It mixes this in with some blocks from the MobileNetv2 architecture, which are used perform downsampling. It uses some cheap but effective data augmentation techniques, such as mixup.

Additionally, to get it to run on the service I was using (an A4000 GPU on paperspace.com), I had to use mixed precision training for memory efficiency, which confers a substantial speed-up too.

The end result is a ~90% success rate in classifying faces from 7001 possible identities. Given that there are some examples in the data where faces are mislabelled, this is pretty good.


## Installation
```
git clone https://github.com/SRagy/FaceClassifier.git
```
Navigate to the install folder and then run

```
pip install -r requirements.txt.
```
Note that the code uses some experimental features from packages such as pytorch-lightning, hence the exact package versions are given in requirements.txt as the implementations of these functions may change in the future. It is hence recommended to run this code in a fresh environment.

The data is obtainable here: https://www.kaggle.com/competitions/11-785-f23-hw2p2-classification/data

To use the default training code, it is expected that the data is in a subdirectory named 'data'.
i.e. 

```
├── FaceClassifer
│   ├── data
│   │   ├── train
│   │   ├── dev
│   │   ├── test
│   ├── architecture.py
│   ├── ...
```

## Data
The dataset used is a relatively small dataset obtained from [this page](https://www.kaggle.com/competitions/11-785-f23-hw2p2-classification/data) on kaggle. It contains 7001 identity labels - i.e. 7001 people to differentiate.

## Project structure
- residual_blocks.py contains the block designs used in the main architecture. 
- architecture.py defines the full architecture by composing the blocks of residual_blocks.py.
- data.py defines the dataloaders and transformations for augmentation and normalisation.
- training.py contains a trainer class for the training process.
- train_example.ipynb and Tests.ipynb are mainly for tinkering with, and hence a bit messy.
- run_training.py runs the training procedure given the data is available.