from __future__ import print_function
import argparse
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from trainer import MAG_NET
from utils import get_config, load_kik, load_korea, load_stead
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt

import os
import seaborn as sns


import smtk.response_spectrum as rsp
import smtk.intensity_measures as ims
from scipy.optimize import curve_fit

from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class MinMaxScaler3D(MinMaxScaler):
    def fit_transform(self, X, y=None):
        x = np.reshape(X, newshape=(X.shape[0]*X.shape[1], X.shape[2]))
        return np.reshape(super().fit_transform(x, y=y), newshape=X.shape)

parser = argparse.ArgumentParser()
model_name = 'MAG_NET'
parser.add_argument('--config', type=str, default='./outputs/config.yaml', help="net configuration")
parser.add_argument('--checkpoint', type=str, default='outputs/checkpoints/Mag_net_00000120', help="checkpoint of autoencoders")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--save_dir', type=str, default='.')
parser.add_argument('--test_size', type=int, default=100)
opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

config = get_config(opts.config)

trainer = MAG_NET(config)
state_dict = torch.load(opts.checkpoint+'.pt')
trainer.model.load_state_dict(state_dict)

trainer.cuda()
trainer.eval()


#DATASET LOAD
dir_name = './test_dataset.npz'
data_file = np.load(dir_name)
seismic_data = data_file['ev']
distance = data_file['dis_cls']
stream_max = data_file['stream_max']
ps_time = data_file['ps_time']
magnitude = data_file['mag']

data_length = len(distance)
print("DATA LENGTH: ", data_length)

#test_dataset = TensorDataset(torch.tensor(seismic_data, dtype=torch.float32).cuda(), torch.tensor(distance, dtype=torch.float32).cuda(), torch.tensor(stream_max, dtype=torch.float32).view(-1, 1).cuda())
test_dataset = TensorDataset(torch.tensor(seismic_data, dtype=torch.float32).cuda(),
                             torch.tensor(distance, dtype=torch.float32).view(-1,1).cuda(),
                             torch.tensor(stream_max, dtype=torch.float32).view(-1,1).cuda(),
                             torch.tensor(ps_time, dtype=torch.float32).view(-1,1).cuda(),
                             torch.tensor(magnitude, dtype=torch.float32).view(-1,1).cuda())
test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

predicted_dis = []
target_dis = []

predicted_mag = []
target_mag = []
for batch_idx, samples in enumerate(test_dataloader):
    test_input_data, test_dis_target, test_stream_max_, test_ps_time, test_magnitude = samples
    test_input_data = torch.transpose(test_input_data, 1, 2)

    predicted_mag_, predicted_dis_ = trainer.model(test_input_data, test_stream_max_, test_ps_time)

    if predicted_dis == []:
        predicted_dis = predicted_dis_.cpu().detach().numpy()
    else:
        predicted_dis = np.append(predicted_dis, predicted_dis_.cpu().detach().numpy(), axis=0)

    if target_dis == []:
        target_dis = test_dis_target.cpu().detach().numpy()
    else:
        target_dis = np.append(target_dis, test_dis_target.cpu().detach().numpy(), axis=0)

    if predicted_mag == []:
        predicted_mag = predicted_mag_.cpu().detach().numpy()
    else:
        predicted_mag = np.append(predicted_mag, predicted_mag_.cpu().detach().numpy(), axis=0)

    if target_mag == []:
        target_mag = test_magnitude.cpu().detach().numpy()
    else:
        target_mag = np.append(target_mag, test_magnitude.cpu().detach().numpy(), axis=0)


#dis_loss = np.mean(np.abs(predicted_dis - target_dis), axis=0)
#mag_loss = np.mean(np.abs(predicted_mag - target_mag), axis=0)

from sklearn.metrics import r2_score

mag_loss = r2_score(target_mag, predicted_mag)
dis_loss = r2_score(target_dis, predicted_dis)


#################### target DIS plot ####################
target_dis = target_dis.reshape(-1)
predicted_dis = predicted_dis.reshape(-1)

fig, ax = plt.subplots()
ax.scatter(target_dis, predicted_dis, alpha=0.05, facecolors='r', edgecolors='r')
ax.plot([target_dis.min(), target_dis.max()], [target_dis.min(), target_dis.max()], 'k--', lw=2)
ax.set_title('Distance')
ax.set_xlabel('True distance (km)')
ax.set_ylabel('Predicted distance (km)')

if 'train' in dir_name:
    fig.text(0, 0, "%.4f" % (dis_loss))
    fig.savefig(os.path.join(opts.save_dir, opts.checkpoint[-6:] + '_train_dis.png'))

else:
    fig.text(0, 0, "%.4f" % (dis_loss))
    fig.savefig(os.path.join(opts.save_dir, opts.checkpoint[-6:] + '_test_dis.png'))


#################### target MAG plot ####################
target_mag = target_mag.reshape(-1)
predicted_mag = predicted_mag.reshape(-1)

fig, ax = plt.subplots()
ax.scatter(target_mag, predicted_mag, alpha=0.05, facecolors='r', edgecolors='r')
ax.plot([target_mag.min(), target_mag.max()], [target_mag.min(), target_mag.max()], 'k--', lw=2)
ax.set_title('Magnitude')
ax.set_xlabel('True magnitude')
ax.set_ylabel('Predicted magnitude')

if 'train' in dir_name:
    fig.text(0, 0, "%.4f" % (mag_loss))
    fig.savefig(os.path.join(opts.save_dir, opts.checkpoint[-6:] + '_train_mag.png'))

else:
    fig.text(0, 0, "%.4f" % (mag_loss))
    fig.savefig(os.path.join(opts.save_dir, opts.checkpoint[-6:] + '_test_mag.png'))