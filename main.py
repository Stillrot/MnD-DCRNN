import numpy as np
import shutil
import os.path
import argparse
from trainer import MAG_NET
from utils import load_kik, get_config, prepare_sub_folder, seis_normalization, evaluation, load_stead, load_korea
import torch.backends.cudnn as cudnn
import torch
import datetime
import time
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from glob import glob
import matplotlib.pyplot as plt


os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
cudnn.benchmark = True

#1.kiknet 2.stead 3.korea
dataset_name = 'stead'

class MinMaxScaler3D(MinMaxScaler):
    def fit_transform(self, X, y=None):
        x = np.reshape(X, newshape=(X.shape[0], X.shape[1] * X.shape[2]))
        return np.reshape(super().fit_transform(x, y=y), newshape=X.shape)


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='seismo.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
opts = parser.parse_args()

is_model = glob('./outputs/checkpoints/*')
if is_model:
    opts.resume = True
else:
    opts.resume = False

is_dataset = glob('./*.npz')

# Load experiment setting
config = get_config(opts.config)
max_epoch = config['max_epoch']

#MODEL SUMMARY
#from torchsummaryX import summary as summary_
#model = Magnitude_Estimation()
#summary_(model.cuda(), torch.zeros(256, 3, 3000, device='cuda'))

trainer = MAG_NET(config)
trainer.cuda()
trainer.train()

model_name = 'MAG_NET'
output_directory = os.path.join(opts.output_path + "/outputs")
checkpoint_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

if opts.resume == False and is_dataset == []:

    if dataset_name == 'kiknet':
        seismic_data = []
        distance = []
        distance_class = []
        ps_time = []
        magnitude = []
        pbar = tqdm(total=11)
        for filenum in range(1, 12):
            pbar.update()
            seismic_data_, distance_, distance_class_, ps_time_, magnitude_ = load_kik(length=config['gen_length'], low_cut=0.5, high_cut=30, file_dir='D:/기상청 제공 KiK-net 데이터/Weather_Metro_KIK_0916_update2/WMKC_{}.npz'.format(filenum))

            if seismic_data == []:
                seismic_data = seismic_data_
            else:
                seismic_data = np.append(seismic_data, seismic_data_, axis=0)

            if distance == []:
                distance = distance_
            else:
                distance = np.append(distance, distance_, axis=0)

            if distance_class == []:
                distance_class = distance_class_
            else:
                distance_class = np.append(distance_class, distance_class_, axis=0)

            if ps_time == []:
                ps_time = ps_time_
            else:
                ps_time = np.append(ps_time, ps_time_, axis=0)

            if magnitude == []:
                magnitude = magnitude_
            else:
                magnitude = np.append(magnitude, magnitude_, axis=0)

        distance = np.array(distance)
        magnitude = np.array(magnitude)
        ps_time = np.array(ps_time)

    if dataset_name == 'stead':
        #STEAD DATASET LOAD
        seismic_data, distance, distance_class, magnitude, ps_time = load_stead(length=config['gen_length'], low_cut=0.5, high_cut=30, file_dir='E:/Seis_Dataset/STEAD/1106dataset_typeall.npz')

        distance = np.array(distance)
        magnitude = np.array(magnitude)
        ps_time = np.array(ps_time)

    if dataset_name == 'korea':
        #KOREA DATASET LOAD
        seismic_data, distance, magnitude, ps_time = load_korea(length=config['gen_length'], low_cut=0.5, high_cut=30, file_dir='D:/Korea_ALL.npz')
        distance = np.array(distance)
        magnitude = np.array(magnitude)
        ps_time = np.array(ps_time)

    #데이터 길이 계산
    all_data_length = len(distance)
    print(all_data_length)


    # stream max 값 계산
    stream_max = []
    for idx in range(all_data_length):
        stream_max.append(np.max(np.abs(seismic_data[idx])))
    stream_max = np.array(stream_max)

    '''
    # 너무 큰 값 5% index 설정
    str_idx = np.argsort(stream_max)
    threshold = -int(all_data_length * 0.05)
    # 데이터 제외
    stream_max = stream_max[str_idx[:threshold]]
    seismic_data = seismic_data[str_idx[:threshold]]
    distance = distance[str_idx[:threshold]]
    ps_time = ps_time[str_idx[:threshold]]
    '''
    # normalization -1~1
    #scaler = MinMaxScaler3D(feature_range=(-1, 1))
    #seismic_data = scaler.fit_transform(seismic_data)
    #seismic_data = (seis_normalization(seismic_data) * 2) - 1


    #plt.plot(seismic_data[550,:,0])
    #plt.savefig(os.path.join('./_1.png'))
    #plt.cla()
    #plt.plot(seismic_data[550,:,1])
    #plt.savefig(os.path.join('./_2.png'))
    #plt.cla()
    #plt.plot(seismic_data[550,:,2])
    #plt.savefig(os.path.join('./_3.png'))
    #scaler2 = MinMaxScaler(feature_range=(-1, 1))
    #stream_max = scaler2.fit_transform(np.reshape(stream_max, (-1, 1)))
    #stream_max = (((stream_max - np.min(stream_max)) / (np.max(stream_max) - np.min(stream_max))) * 2) - 1
    #stream_max = np.reshape(stream_max, (-1, 1))

    #scaler3 = MinMaxScaler(feature_range=(-1, 1))
    #ps_time = scaler3.fit_transform(np.reshape(ps_time, (-1, 1)))
    #ps_time = (ps_time - np.min(ps_time)) / (np.max(ps_time) - np.min(ps_time)) #0~1

    '''
    ps_time_x = []
    for t_idx in range(len(ps_time)):
        tmp = np.zeros(5)
        tmp[int(ps_time[t_idx]/6)] = 1
        ps_time_x.append(tmp)
    ps_time_x = np.array(ps_time_x)'''

    # 데이터 셔플
    s = np.arange(seismic_data.shape[0])
    np.random.shuffle(s)
    seismic_data = seismic_data[s]
    distance = distance[s]
    stream_max = stream_max[s]
    ps_time_x = ps_time[s]
    magnitude = magnitude[s]

    #train/test 데이터 나누기
    training_seis = seismic_data[:int(0.8 * len(seismic_data))]
    test_seis = seismic_data[int(0.8 * len(seismic_data)):]

    training_dis_cls = distance[:int(0.8 * len(distance))]
    test_dis_cls = distance[int(0.8 * len(distance)):]

    training_stream_max = stream_max[:int(0.8 * len(stream_max))]
    test_stream_max = stream_max[int(0.8 * len(stream_max)):]

    training_ps_time = ps_time_x[:int(0.8 * len(ps_time_x))]
    test_ps_time = ps_time_x[int(0.8 * len(ps_time_x)):]

    training_magnitude = magnitude[:int(0.8 * len(magnitude))]
    test_magnitude = magnitude[int(0.8 * len(magnitude)):]

    train_file_name = './train_dataset'
    test_file_name = './test_dataset'

    np.savez(train_file_name, ev=training_seis, dis_cls=training_dis_cls, stream_max=training_stream_max, ps_time=training_ps_time, mag=training_magnitude)
    np.savez(test_file_name, ev=test_seis, dis_cls=test_dis_cls, stream_max=test_stream_max, ps_time=test_ps_time, mag=test_magnitude)

else:
    data_file = np.load('./train_dataset.npz')
    training_seis = data_file['ev']
    training_dis_cls = data_file['dis_cls']
    training_stream_max = data_file['stream_max']
    training_ps_time = data_file['ps_time']
    training_magnitude = data_file['mag']


data_length = len(training_dis_cls)
print("TRAINING DATA LENGTH: ", data_length)

dataset = TensorDataset(torch.tensor(training_seis, dtype=torch.float32).cuda(),
                        torch.tensor(training_dis_cls, dtype=torch.float32).view(-1,1).cuda(),
                        torch.tensor(training_stream_max, dtype=torch.float32).view(-1,1).cuda(),
                        torch.tensor(training_ps_time, dtype=torch.float32).view(-1,1).cuda(),
                        torch.tensor(training_magnitude, dtype=torch.float32).view(-1,1).cuda())
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

epoch = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
start_time = time.time()

eval_data_file = np.load('./test_dataset.npz')

for i in range(epoch, max_epoch+1):
    for batch_idx, samples in enumerate(dataloader):
        input_data, dis_target, stream_max_, target_ps_time, target_magnitude = samples
        input_data = torch.transpose(input_data, 1, 2)
        dis_loss = trainer.model_update(input_data, dis_target.cuda(), stream_max_, target_ps_time.cuda(), target_magnitude.cuda(), i, batch_idx)

        elapsed = str(datetime.timedelta(seconds=time.time() - start_time))[:-7]
        if batch_idx % 400 == 0:
            print('Elapsed time {}  Epoch {:4d}/{} Batch {}/{} Loss: {:.6f}'.format(elapsed, i + 1, max_epoch, batch_idx + 1, len(dataloader), dis_loss.item()))

    # Save network weights
    if (epoch + 1) % config['snapshot_save_iter'] == 0:
        trainer.save(checkpoint_directory, epoch)

    epoch += 1
    if epoch >= max_epoch:
        print('Finish training')

    trainer.model_scheduler.step(dis_loss)

    if i % 1 == 0:
        evaluation(trainer.model, eval_data_file, i)