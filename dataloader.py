from sklearn.utils import shuffle
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from config import args

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

def fetch_dataloader():

    # Change the dimension, change to tensor between 0 and 1, shift t0 -1 and 1
    transform_train = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                                      transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])

    transform_val = transforms.Compose([transforms.Resize((224,224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])

    training_dataset = datasets.ImageFolder('data/train', transform=transform_train)
    validation_dataset = datasets.ImageFolder('data/val', transform=transform_val)

    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=args.train_batch, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size =args.val_batch, shuffle=False)

    return training_loader, validation_loader


if __name__ == '__main__':
  # Check the data
  training_loader, _ = fetch_dataloader()
  dataiter = iter(training_loader)
  images, labels = dataiter.next()
  print('Dataloader:')
  print(f'  Batch size: {images.shape[0]}')
  print(f'  Input shape: {images.shape}')
  print(f'  Label shape: {labels.shape}')

  # Convert to something that could be visualized
  def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)
    return image

  # Visualizing a batch
  fig = plt.figure(figsize=(25, 4))
  for idx in np.arange(10):
    ax = fig.add_subplot(2, 5, idx+1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title(labels[idx].item())
  plt.pause(0.05)
  plt.show()