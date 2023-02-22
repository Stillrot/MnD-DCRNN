import os
import math
from scipy.signal import butter, lfilter
import yaml
import numpy as np
import torch.nn.init as init
from tqdm import tqdm
import torch
from munch import Munch
import matplotlib.pyplot as plt
import pandas as pd
import h5py

def load_korea(length, low_cut, high_cut, file_dir):
    data_file = np.load(file_dir)
    dataA = data_file['ev']
    dis_corpus = data_file['dis']
    mag_corpus = data_file['mag']
    p_pick_corpus = data_file['p_pick']
    s_pick_corpus = data_file['s_pick']

    data = []
    dis = []
    dis_cls = []
    mag = []
    pbar = tqdm(total=len(dataA))
    for k in range(len(dataA)):
        pbar.update()
        data_ = dataA[k][np.newaxis, int(p_pick_corpus[k]*100)-500:int(p_pick_corpus[k]*100)-500+length, :]
        if data_.shape[1] == length:
            data.append(data_)
            dis.append(dis_corpus[k])
            mag.append(mag_corpus[k])
            #dis_cls.append(dist_class(dis_corpus[k]))
    dataA = np.concatenate(data, axis=0).astype('float32')

    # high pass filter
    data = []

    for i in range(len(dataA)):
        filtered_data = butter_bandpass_filter(dataA[i], low_cut, high_cut, fs=100)
        data.append(np.expand_dims(filtered_data, axis=0))
    data = np.concatenate(data, axis=0).astype('float32')
    return data, dis, mag, mag


def load_stead(length, low_cut, high_cut, file_dir):
    data_file = np.load(file_dir)
    dataA = data_file['ev']
    dis_corpus = data_file['dis']
    mag_corpus = data_file['mag']
    p_pick_corpus = data_file['p_pick']
    s_pick_corpus = data_file['s_pick']

    data = []
    dis = []
    dis_cls = []
    mag = []
    pbar = tqdm(total=len(dataA))
    for k in range(len(dataA)):
        pbar.update()
        if dis_corpus[k] <= 120:
            data_ = dataA[k][np.newaxis, int(p_pick_corpus[k])-500:int(p_pick_corpus[k])-500+length, :]
            data.append(data_)
            dis.append(dis_corpus[k])
            mag.append(mag_corpus[k])
            dis_cls.append(dist_class(dis_corpus[k]))
    dataA = np.concatenate(data, axis=0).astype('float32')

    # high pass filter
    data = []

    for i in range(len(dataA)):
        filtered_data = butter_bandpass_filter(dataA[i], low_cut, high_cut, fs=100)
        data.append(np.expand_dims(filtered_data, axis=0))
    data = np.concatenate(data, axis=0).astype('float32')
    return data, dis, dis_cls, mag, mag

def load_kik(length, low_cut, high_cut, file_dir):
    data_file = np.load(file_dir)
    dataA = data_file['ev']
    dis_corpus = data_file['dis']
    mag_corpus = data_file['mag']
    ps_time_corpus = data_file['ps_time']

    data = []
    dis = []
    dis_cls = []
    ps_time = []
    mag = []
    pbar = tqdm(total=len(dataA))
    for k in range(len(dataA)):
        pbar.update()
        if snr_cal(dataA[k]) >= 5 and dis_corpus[k] <= 120:
            data_ = dataA[k][np.newaxis, :length, :]
            data.append(data_)
            dis.append(dis_corpus[k])
            mag.append(mag_corpus[k])
            dis_cls.append(dist_class(dis_corpus[k]))
            ps_time.append(ps_time_corpus[k])
    dataA = np.concatenate(data, axis=0).astype('float32')

    # high pass filter
    data = []

    for i in range(len(dataA)):
        filtered_data = butter_bandpass_filter(dataA[i], low_cut, high_cut, fs=100)
        data.append(np.expand_dims(filtered_data, axis=0))
    data = np.concatenate(data, axis=0).astype('float32')
    return data, dis, dis_cls, ps_time, mag

def dist_class(dist):
    if dist>=0 and dist<20:
        label = [1,0,0,0,0,0]
        #label = 0
    elif dist>=20 and dist<40:
        label = [0,1,0,0,0,0]
        #label = 1
    elif dist>=40 and dist<60:
        label = [0,0,1,0,0,0]
        #label = 2
    elif dist>=60 and dist<80:
        label = [0,0,0,1,0,0]
        #label = 3
    elif dist>=80 and dist<100:
        label = [0,0,0,0,1,0]
        #label = 4
    elif dist>=100 and dist<=120:
        label = [0,0,0,0,0,1]
        #label = 5

    return label

def snr_cal(data, p_pick=500, margin=20):
    data = data[:1000, 2]
    noise = data[:p_pick-margin]
    signal = data[p_pick+margin:]
    ratio = np.linalg.norm(signal, ord=2, axis=0)/np.linalg.norm(noise, ord=2, axis=0)
    snr = 10*np.log10(ratio)
    return snr

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def prepare_sub_folder(output_directory):
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory

def butter_bandpass(lowcut, high_cut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, high_cut, fs, order=5):
    b, a = butter_bandpass(lowcut, high_cut, fs, order=order)
    y_ = []
    for i in range(3):
        y = lfilter(b, a, data[:, i])
        y_.append(np.expand_dims(y, axis=1))
    y_ = np.concatenate(y_, axis=1)
    return y_

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun

def seis_normalization(seismic_data_):
    seismic_data = []
    for i in range(seismic_data_.shape[0]):
        std = np.std(seismic_data_[i], axis=0)
        if std[0]!=0 and std[1]!=0 and std[2]!=0:
            # standard score
            e = seismic_data_[i, :, 0]
            n = seismic_data_[i, :, 1]
            z = seismic_data_[i, :, 2]
            e_hat = (e-np.mean(e)+0.00001)/np.std(e)
            n_hat = (n-np.mean(n)+0.00001)/np.std(n)
            z_hat = (z-np.mean(z)+0.00001)/np.std(z)
            e_minmax = (e_hat-np.min(e_hat))/(np.max(e_hat)-np.min(e_hat))
            n_minmax = (n_hat - np.min(n_hat))/ (np.max(n_hat) - np.min(n_hat))
            z_minmax = (z_hat - np.min(z_hat))/ (np.max(z_hat) - np.min(z_hat))

            seismic_data.append(np.concatenate([e_minmax[np.newaxis, :, np.newaxis],
                                                n_minmax[np.newaxis, :, np.newaxis],
                                                z_minmax[np.newaxis, :, np.newaxis]], axis=2))

    seismic_data = np.concatenate(seismic_data, axis=0)
    return seismic_data


def string_convertor(dd):
    dd2 = dd.split()
    SNR = []
    for i, d in enumerate(dd2):
        if d != '[' and d != ']':

            dL = d.split('[')
            dR = d.split(']')

            if len(dL) == 2:
                dig = dL[1]
            elif len(dR) == 2:
                dig = dR[0]
            elif len(dR) == 1 and len(dR) == 1:
                dig = d
            try:
                dig = float(dig)
            except Exception:
                dig = None

            SNR.append(dig)
    return (SNR)

def string_convertor_coda(dd):
    dd = float(dd[2:-3])
    return (dd)


from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
@torch.no_grad()
def evaluation(model, data_file, epoch):
    # DATASET LOAD
    #data_file = np.load('./test_dataset.npz')
    seismic_data = data_file['ev']
    distance = data_file['dis_cls']
    stream_max = data_file['stream_max']
    ps_time = data_file['ps_time']
    mag = data_file['mag']

    data_length = len(distance)

    # test_dataset = TensorDataset(torch.tensor(seismic_data, dtype=torch.float32).cuda(), torch.tensor(distance, dtype=torch.float32).cuda(), torch.tensor(stream_max, dtype=torch.float32).view(-1, 1).cuda())
    test_dataset = TensorDataset(torch.tensor(seismic_data, dtype=torch.float32).cuda(),
                                 torch.tensor(distance, dtype=torch.float32).view(-1,1).cuda(),
                                 torch.tensor(stream_max, dtype=torch.float32).view(-1,1).cuda(),
                                 torch.tensor(ps_time, dtype=torch.float32).view(-1,1).cuda(),
                                 torch.tensor(mag, dtype=torch.float32).view(-1,1).cuda())
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    predicted_dis = []
    target_dis = []
    predicted_mag = []
    target_mag = []
    for batch_idx, samples in enumerate(test_dataloader):
        test_input_data, test_dis_target, test_stream_max_, test_ps_time, test_mag = samples
        test_input_data = torch.transpose(test_input_data, 1, 2)

        predicted_mag_, predicted_dis_ = model(test_input_data, test_stream_max_, test_ps_time)

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
            target_mag = test_mag.cpu().detach().numpy()
        else:
            target_mag = np.append(target_mag, test_mag.cpu().detach().numpy(), axis=0)

    dis_loss = np.mean(np.abs(predicted_dis - target_dis), axis=0)
    mag_loss = np.mean(np.abs(predicted_mag - target_mag), axis=0)
    print("=================%.5f================%.5f===============" % (dis_loss, mag_loss))

    target = target_dis.reshape(-1)
    predicted = predicted_dis.reshape(-1)

    fig, ax = plt.subplots()
    ax.scatter(target, predicted, alpha=0.05, facecolors='r', edgecolors='r')
    ax.plot([target.min(), target.max()], [target.min(), target.max()], 'k--', lw=2)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    fig.savefig('./loss_dis/{}_1.png'.format(epoch))

    target = target_mag.reshape(-1)
    predicted = predicted_mag.reshape(-1)

    fig, ax = plt.subplots()
    ax.scatter(target, predicted, alpha=0.05, facecolors='r', edgecolors='r')
    ax.plot([target.min(), target.max()], [target.min(), target.max()], 'k--', lw=2)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    fig.savefig('./loss_mag/{}_1.png'.format(epoch))


    '''
    str_idx = np.argsort(target)
    threshold = -int(len(target) * 0.005)
    # 데이터 제외
    predicted = predicted[str_idx[:threshold]]
    target = target[str_idx[:threshold]]
    '''