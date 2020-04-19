# @Author: Ibrahim Salihu Yusuf <Ibrahim>
# @Date:   2019-12-10T12:28:39+02:00
# @Email:  sibrahim1396@gmail.com
# @Project: Audio Classifier
# @Last modified by:   yusuf
# @Last modified time: 2019-12-21T09:50:23+02:00

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import moviepy.editor as mp
import torchvision
from  keras import backend as K
from keras.models import load_model, Model
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split

from models2 import *
from utils2 import *
import json, time

train_file_path = "/home/jupyter/deepfake/train_sample_videos/metadata.json"

df = pd.read_json(train_file_path).transpose()
df['label'] = le.fit_transform(df.label.values)
train_df, test_df = train_test_split(df, test_size=0.2)

video_paths = "/home/jupyter/deepfake/train_sample_videos"

model = load_model("saved_model_240_8_32_0.05_1_50_0_0.0001_100_156_2_True_True_fitted_objects.h5.1", custom_objects={'customPooling': customPooling})
audio_model = Model(inputs = model.input, outputs = model.layers[-3].output)

frame_model = torch.hub.load('pytorch/vision:v0.4.2', 'inception_v3', pretrained=True)
frame_model.eval()
frame_model.fc = Identity()

img_transforms = transforms.Compose([transforms.Resize((229, 229)),
                                 transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

avg_loss = []
overall_result = {}

def main():
    model = DeepFake().to(device).apply(init_weights)
    model_name = "DeepFake_model"

    writer_path = "/results/{}".format(model_name)
    # comment = "{}".format(models_name[model_idx])
#     writer = SummaryWriter(writer_path)
    inter_result = []

    train_dataset = VideoDataset(df=train_df, video_paths=video_paths, frame_model=frame_model, audio_model=audio_model, transform=img_transforms, limit=150)
    test_dataset = VideoDataset(df=test_df, video_paths=video_paths, frame_model=frame_model, audio_model=audio_model, transform=img_transforms, limit=150)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.007645)

    early_stopping = EarlyStopping(patience=5, verbose=True)

    if os.path.isfile('DeepFake_model_checkpoint.tar'):
        print('DeepFake_model_checkpoint.tar found..')
        print('Loading checkpoint..')
        checkpoint = torch.load('DeepFake_model_checkpoint.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Finished loading checkpoint..')

    epochs = 2
    print_every = 1

    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []

    for i in range(epochs):
      print("Start of Training epoch {}".format(i+1))
      t0 = time.time()
      train_loss, train_acc = train(model, train_loader, optimizer, criterion)
      t1 = time.time()
      print("Train Loss at {}/{} is {} | Train accuracy:{} | Time:{}".format(i+1, epochs, train_loss, train_acc, t1-t0))

      train_losses.append(train_loss)
      train_accuracy.append(train_acc)

#       writer.add_scalar("Fold{}/Loss/Train".format(i), train_loss, i)
#       writer.add_scalar("Fold{}/Accuracy/Train".format(i), train_acc, i)

      if (i+1)%print_every == 0:
          print("Start of testing epoch {}".format(i+1))
          t0 = time.time()
          test_loss, test_acc = test(model, test_loader, criterion)
          t1 = time.time()
          print("Test Loss at {}/{} is {} | Test accuracy:{} | Time:{}".format(i+1, epochs, train_loss, train_acc, t1-t0))

          test_losses.append(test_loss)
          test_accuracy.append(test_acc)

#           writer.add_scalar("Fold{}/Loss/Test".format(i), test_loss, i)
#           writer.add_scalar("Fold{}/Accuracy/Test".format(i), test_acc, i)

          early_stopping(test_loss, model)

          if early_stopping.early_stop:
             print("Early stopping")
             break

      scheduler.step()

    inter_result.append({'train_loss':train_losses, 'train_accuracy':train_accuracy, 'test_loss':test_losses, 'test_accuracy':test_accuracy})
    overall_result["{}".format(model_name)] = inter_result
    out_filename = '/results/{}_result.json'.format(model_name)
    writer.close()
    try:
        with open(out_filename, 'w') as f:
            json.dump(inter_result, f)
    except:
        pass
if __name__ == "__main__":
    t0 = time.time()
    main()
    t1 = time.time()
    print("Total Training time is: {}".format(t1-t0))
    print("\n\nEnd of training\nLogging results..")
    with open('/data/overall_results.json', 'w') as f:
        json.dump(overall_result, f)
