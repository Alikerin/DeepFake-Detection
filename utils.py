# @Author: Ibrahim Salihu Yusuf <Ibrahim>
# @Date:   2019-12-10T11:24:45+02:00
# @Email:  sibrahim1396@gmail.com
# @Project: Audio Classifier
# @Last modified by:   yusuf
# @Last modified time: 2019-12-21T09:49:37+02:00



import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch.nn as nn
from  keras import backend as K
from sklearn import preprocessing
import moviepy.editor as mp
import torchvision
from keras.models import load_model, Model
import torchvision.transforms as transforms
from PIL import Image


le = preprocessing.LabelEncoder()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VideoDataset(Dataset):
    """
    A rapper class for the Video dataset.
    """

    def __init__(self, video_paths, frame_model, audio_model, transform = None, limit=None, file_path=None, df=None):
        """
        Args:
            file_path(string): path to the audio csv file
            root_dir(string): directory with all the audio folds
            folds: integer corresponding to audio fold number or list of fold number if more than one fold is needed
        """
        self.video_file = df
#         elif file_path:
#             self.video_file = pd.read_json(file_path).transpose()
#             self.video_file['label'] = le.fit_transform(self.video_file.label.values)
        # self.folds = folds
        # self.video_paths = glob.glob(video_paths + '/*' + str(self.folds) + '/*')
        self.video_paths = video_paths
        self.frame_model = frame_model.eval()
        self.audio_model = audio_model
        self.transform = transform
        self.limit = limit

    def __len__(self):
        return len(self.video_file)-1

    def __getitem__(self, idx):

        video_file = self.video_file.index[idx]
        clip = mp.VideoFileClip(os.path.join(self.video_paths, video_file)) #pass in the video file path here
        no_frames = int(np.round(clip.duration)) * int(np.round(clip.fps))

        if self.limit:
            frame_embeddings = np.zeros((self.limit, 2048))
        else:
            frame_embeddings = np.zeros((no_frames, 2048))

        for j, frame in enumerate(clip.iter_frames()):
            if j == self.limit:
                break
            frame = Image.fromarray(frame)
            if self.transform:
                frame = self.transform(frame)
            frame_embeddings[j, :] = (self.frame_model(frame.unsqueeze(0)).detach().numpy()).squeeze(0) #detach from gradients

        audio = clip.audio.to_soundarray()

        audio = audio.mean(1, keepdims=True)
        mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate=44100, n_fft=2048, win_length=2000, hop_length=500, n_mels=240)(torch.tensor(audio).float().T)  # (channel, n_mels, time)
        mel_specgram = np.moveaxis(mel_specgram.detach().numpy(), 1, 2)
        audio_embedding = [self.audio_model.predict(mel_specgram).squeeze()]
        label = self.video_file['label'][idx]

        return torch.tensor(frame_embeddings), torch.tensor(audio_embedding), torch.tensor(label)

def init_weights(m):
    if type(m) == nn.Conv1d or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight.data)

def train(model, train_loader, optimizer, criterion):
  print("Training Model")
  model.train()
  train_loss = 0
  train_correct = 0
  for frame_data, audio_data, label in train_loader:
    frame_data = frame_data.to(device)
    label = label.to(device)
    audio_data = audio_data.to(device)
    optimizer.zero_grad()
    out = model(frame_data, audio_data)
    train_correct += (torch.argmax(out, dim=1).eq_(label).sum()).item()
    loss = criterion(out, label)
    train_loss += loss.item()
    loss.backward()
    optimizer.step()
  avg_loss = train_loss/len(train_loader)
  accuracy = train_correct/(len(train_loader.dataset))
  return avg_loss, accuracy

def test(model, test_loader, criterion):
  print("Testing Model")
  with torch.no_grad():
    model.eval()
    test_correct = 0
    test_loss = 0
    for frame_data, audio_data, label in test_loader:
        frame_data = frame_data.to(device)
        label = label.to(device)
        audio_data = audio_data.to(device)
        out = model(frame_data, audio_data)
        loss2 = criterion(out2, label)
        test_loss += loss2.item()
        test_correct += (torch.argmax(out2, dim=1).eq_(label).sum()).item()
    avg_loss = test_loss/len(test_loader)
    accuracy = test_correct/len(test_loader.dataset)
  return avg_loss, accuracy


def customPooling(x):
    target = x[1]
    inputs = x[0]
    maskVal = 0
    #getting the mask by observing the model's inputs
    mask = K.equal(inputs, maskVal)
    mask = K.all(mask, axis=-1, keepdims=True)

    #inverting the mask for getting the valid steps for each sample
    mask = 1 - K.cast(mask, K.floatx())

    #summing the valid steps for each sample
    stepsPerSample = K.sum(mask, axis=1, keepdims=False)

    #applying the mask to the target (to make sure you are summing zeros below)
    target = target * mask

    #calculating the mean of the steps (using our sum of valid steps as averager)
    means = K.sum(target, axis=1, keepdims=False) / stepsPerSample

    return means

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased ({} --> {}).  Saving model ...'.format(self.val_loss_min, val_loss))

        DeepFake_model_checkpoint={'model_state_dict':model.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                        }

        torch.save(DeepFake_model_checkpoint, 'DeepFake_model_checkpoint.tar')
        self.val_loss_min = val_loss
