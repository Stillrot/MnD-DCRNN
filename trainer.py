import torch
import torch.nn as nn
import os
from utils import weights_init, get_model_list
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

class MAG_NET(nn.Module):
    def __init__(self, hyperparameters):
        super(MAG_NET, self).__init__()
        self.model = Magnitude_Estimation()
        lr = hyperparameters['lr']
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        model_params = list(self.model.parameters())
        self.model_opt = torch.optim.Adam([p for p in model_params if p.requires_grad],
                                          lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.model_scheduler = ReduceLROnPlateau(self.model_opt, factor=np.sqrt(0.1), cooldown=0, min_lr=0.5e-6, patience=20, verbose=True)
        self.apply(weights_init(hyperparameters['init']))

    def forward(self, x):
        pass

    def model_update(self, wave_form, dis_target, max_value, ps_time, mag, iteration, batch_idx):
        self.model_opt.zero_grad()
        pred_mag, pred_dis = self.model(wave_form, max_value, ps_time)
        loss_mag = torch.mean(torch.abs(mag - pred_mag), axis=0)
        loss_dis = torch.mean(torch.abs(dis_target - pred_dis), axis=0)
        loss = 100*loss_mag + loss_dis

        #loss = F.cross_entropy(pred_distance, torch.argmax(dis_target, 1))
        loss.backward()
        self.model_opt.step()
        #self.model_scheduler.step(loss)
        return loss

    def save(self, dir, iterations):
        model_name = os.path.join(dir, 'Mag_net_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(dir, 'optimizer.pt')

        torch.save(self.model.state_dict(), model_name)
        torch.save({'Mag_net': self.model_opt.state_dict()}, opt_name)

    def resume(self, checkpoint_dir, hyperparameters):
        # Load
        last_model_name = get_model_list(checkpoint_dir, "Mag_net")
        state_dict = torch.load(last_model_name)
        self.model.load_state_dict(state_dict)
        iterations = int(last_model_name[-11:-3])

        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.model_opt.load_state_dict(state_dict['Mag_net'])

        # Reinitilize schedulers
        self.model_scheduler = ReduceLROnPlateau(self.model_opt, factor=np.sqrt(0.1), cooldown=0, min_lr=0.5e-6, patience=20, verbose=True)
        print('Resume from iteration %d' % iterations)
        return iterations


class Magnitude_Estimation(nn.Module):
    def __init__(self):
        super(Magnitude_Estimation, self).__init__()

        self.filters = [32, 64, 96, 128, 256]

        self.conv0 = spectral_norm(nn.Conv1d(in_channels=3, out_channels=self.filters[3], kernel_size=9, padding=1))
        self.conv1 = spectral_norm(nn.Conv1d(in_channels=self.filters[3], out_channels=self.filters[3], kernel_size=9, padding=1))
        self.conv2 = spectral_norm(nn.Conv1d(in_channels=self.filters[3], out_channels=self.filters[3], kernel_size=9, padding=1))
        self.conv3 = spectral_norm(nn.Conv1d(in_channels=self.filters[3], out_channels=self.filters[3], kernel_size=9, padding=1))


        self.conv4 = spectral_norm(nn.Conv1d(in_channels=self.filters[3], out_channels=self.filters[3], kernel_size=7, padding=1))
        self.conv5 = spectral_norm(nn.Conv1d(in_channels=self.filters[3], out_channels=self.filters[3], kernel_size=7, padding=1))
        self.conv6 = spectral_norm(nn.Conv1d(in_channels=self.filters[3], out_channels=self.filters[3], kernel_size=7, padding=1))
        self.conv7 = spectral_norm(nn.Conv1d(in_channels=self.filters[3], out_channels=self.filters[3], kernel_size=7, padding=1))

        self.conv8 = spectral_norm(nn.Conv1d(in_channels=self.filters[3], out_channels=self.filters[3], kernel_size=5, padding=1))
        self.conv9 = spectral_norm(nn.Conv1d(in_channels=self.filters[3], out_channels=self.filters[3], kernel_size=5, padding=1))
        self.conv10 = spectral_norm(nn.Conv1d(in_channels=self.filters[3], out_channels=self.filters[3], kernel_size=5, padding=1))
        self.conv11 = spectral_norm(nn.Conv1d(in_channels=self.filters[3], out_channels=self.filters[3], kernel_size=5, padding=1))

        self.conv12 = spectral_norm(nn.Conv1d(in_channels=self.filters[3], out_channels=self.filters[3], kernel_size=3, padding=1))
        self.conv13 = spectral_norm(nn.Conv1d(in_channels=self.filters[3], out_channels=self.filters[3], kernel_size=3, padding=1))
        self.conv14 = spectral_norm(nn.Conv1d(in_channels=self.filters[3], out_channels=self.filters[3], kernel_size=3, padding=1))
        self.conv15 = spectral_norm(nn.Conv1d(in_channels=self.filters[3], out_channels=self.filters[3], kernel_size=3, padding=1))

        self.Maxpooling0 = nn.MaxPool1d(kernel_size=2, padding=1)
        self.Maxpooling1 = nn.MaxPool1d(kernel_size=2, padding=1)
        self.Maxpooling2 = nn.MaxPool1d(kernel_size=2, padding=1)
        self.Maxpooling3 = nn.MaxPool1d(kernel_size=2, padding=1)

        self.drop02 = nn.Dropout(p=0.2)
        self.drop4 = nn.Dropout(p=0.5)
        self.drop5 = nn.Dropout(p=0.5)


        self.LSTM00 = nn.LSTM(input_size=252, hidden_size=100, dropout=0.0, bidirectional=True, batch_first=True)
        self.LSTM10 = nn.LSTM(input_size=252, hidden_size=100, dropout=0.0, bidirectional=True, batch_first=True)
        self.linear00 = nn.Linear(in_features=200+1, out_features=128)
        self.linear01 = nn.Linear(in_features=128, out_features=128)
        self.linear02 = nn.Linear(in_features=128, out_features=1)

        self.linear10 = nn.Linear(in_features=200+1, out_features=128)
        self.linear11 = nn.Linear(in_features=128, out_features=128)
        self.linear12 = nn.Linear(in_features=128, out_features=1)

        self.lrelu = nn.LeakyReLU(0.2)

        self.attn1 = Self_Attn(200+1, 'relu')
        self.attn2 = Self_Attn(self.filters[1], 'relu')

        self.batchnorm1 = nn.BatchNorm1d(self.filters[3])
        self.batchnorm2 = nn.BatchNorm1d(self.filters[3])
        self.batchnorm3 = nn.BatchNorm1d(self.filters[3])
        self.batchnorm4 = nn.BatchNorm1d(self.filters[3])

        self.relu0 = nn.ReLU()
        self.relu1 = nn.ReLU()

        self.fc_max = nn.Sequential(nn.Linear(1, 16), nn.LeakyReLU(),
                                    nn.Linear(16, 16), nn.LeakyReLU(),
                                    nn.Linear(16, 16), nn.LeakyReLU(),
                                    nn.Linear(16, 1), nn.LeakyReLU())

        self.layer_norm01 = nn.LayerNorm(200 + 1, eps=1e-05, elementwise_affine=True)
        self.layer_norm02 = nn.LayerNorm(200 + 1, eps=1e-05, elementwise_affine=True)

    def forward(self, x, max_value, ps_time):
        max_value_emb = self.fc_max(max_value)

        x = self.conv0(x)
        x = self.lrelu(x)
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.drop02(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.conv3(x)
        x = self.Maxpooling0(x)
        x = self.lrelu(x)
        x = self.drop02(x)


        x = self.conv4(x)
        x = self.lrelu(x)
        x = self.conv5(x)
        x = self.lrelu(x)
        x = self.drop02(x)
        x = self.conv6(x)
        x = self.lrelu(x)
        x = self.conv7(x)
        x = self.Maxpooling1(x)
        x = self.lrelu(x)
        x = self.drop02(x)

        x = self.conv8(x)
        x = self.lrelu(x)
        x = self.conv9(x)
        x = self.lrelu(x)
        x = self.drop02(x)
        x = self.conv10(x)
        x = self.lrelu(x)
        x = self.conv11(x)
        x = self.Maxpooling2(x)
        x = self.lrelu(x)
        x = self.drop02(x)

        x = self.conv12(x)
        x = self.lrelu(x)
        x = self.conv13(x)
        x = self.lrelu(x)
        x = self.drop02(x)
        x = self.conv14(x)
        x = self.lrelu(x)
        x = self.conv15(x)
        x = self.Maxpooling3(x)
        x = self.lrelu(x)
        cnn_x = self.drop02(x)


        lstm_x1, _ = self.LSTM10(cnn_x)
        x_norm1 = self.layer_norm01(torch.cat([lstm_x1[:, -1, :] * max_value_emb, max_value_emb], 1))
        x = self.linear10(x_norm1)
        x = self.relu0(x)
        x = self.drop4(x)

        x = self.linear11(x)
        x = self.relu1(x)
        x = self.drop5(x)
        dis_x = self.linear12(x)


        #lstm_x0, _ = self.LSTM00(cnn_x)
        x_norm0 = self.layer_norm02(torch.cat([lstm_x1[:, -1, :] * max_value_emb, max_value_emb], 1))

        x = self.linear00(x_norm0)
        x = self.relu0(x)
        x = self.drop4(x)
        x = self.linear01(x)
        x = self.relu1(x)
        x = self.drop5(x)
        mag_x = self.linear02(x)

        # torch.cat((x_norm1, mag_x), 1)

        return mag_x, dis_x

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X L)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Length)
        """
        m_batchsize, C, length = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, length).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, length)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, length)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, length)

        out = self.gamma * out + x
        return out
