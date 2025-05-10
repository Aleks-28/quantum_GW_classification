from numpy import indices
import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from dataset import *
import mlflow


class Encoder(nn.Module):
    """
    :param in_channels_number: Number of input channels.
    :param last_channel_size: Number of output channels.
    :param out_dim: Dimensionality of latent representation.
    :param act_fun: Activation function.
    """
    def __init__(self, 
    in_channels_number, 
    last_channel_size, 
    out_dim,
    act_fun,
    conv1_size, 
    conv2_size,
    conv3_size,
    kernel_size,
    pool_size, 
    linear_size,
    last_signal_size,
    dropout):

        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels_number, conv1_size, kernel_size), 
            nn.BatchNorm1d(conv1_size),
            act_fun(),
            nn.Conv1d(conv1_size, conv2_size, kernel_size),
            nn.BatchNorm1d(conv2_size),   
            act_fun())

        self.pool1 = nn.MaxPool1d(pool_size, return_indices=True)

        self.conv2 = nn.Sequential(
            nn.Conv1d(conv2_size,conv3_size,kernel_size),
            nn.BatchNorm1d(conv3_size),
            act_fun(),
            nn.Conv1d(conv3_size, last_channel_size, kernel_size),
            nn.BatchNorm1d(last_channel_size),
            act_fun(),
            nn.Dropout(dropout)
            )
        
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(last_channel_size * last_signal_size, linear_size),
            act_fun(),
            nn.Linear(linear_size,out_dim))

    def forward(self, x):
        x = self.conv1(x)
        x,indices = self.pool1(x)
        x = self.conv2(x)
        x = self.linear(x)
        return x,indices
        
        
class Decoder(nn.Module):
    """
    :param in_channels_number: Number of input channels.
    :param last_channel_size: Number of output channels.
    :param out_dim: Dimensionality of latent representation.
    :param act_fun: Activation function.
    """
    def __init__(self, 
    in_channels_number, 
    last_channel_size, 
    out_dim,
    act_fun,
    conv1_size, 
    conv2_size,
    conv3_size,
    kernel_size,
    pool_size, 
    linear_size,
    last_signal_size):

        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(out_dim,linear_size),
            act_fun(),
            nn.Linear(linear_size ,last_channel_size * last_signal_size),
            act_fun()
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(last_channel_size,conv3_size,kernel_size),
            nn.BatchNorm1d(conv3_size),
            act_fun(),
            nn.ConvTranspose1d(conv3_size, conv2_size, kernel_size),
            nn.BatchNorm1d(conv2_size),
            act_fun())

        self.unpool1 = nn.MaxUnpool1d(pool_size)

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(conv2_size,conv1_size,kernel_size),
            nn.BatchNorm1d(conv1_size),
            act_fun(),
            nn.ConvTranspose1d(conv1_size,in_channels_number,kernel_size)
        )
        

    def forward(self, x, indices, last_channel_size, last_signal_size):
        x = self.linear(x)
        x = x.reshape(x.shape[0], last_channel_size, last_signal_size)
        x = self.deconv1(x)
        x = self.unpool1(x,indices)
        x = self.deconv2(x)
        return x

        
class AutoEncoder(pl.LightningModule):
    """
    :param in_channels_number: Number of input channels.
    :param last_channel_size: Number of output channels.
    :param out_dim: Dimensionality of latent representation.
    :param act_fun: Activation function.
    """      
    def __init__(self, 
    in_channels_number : int = 1, 
    last_channel_size: int = 64, 
    out_dim : int = 32, 
    act_fun: object = nn.ReLU,
    conv1_size: int = 256, 
    conv2_size: int = 128,
    conv3_size: int = 64,
    kernel_size: int = 3,
    pool_size: int = 2, 
    linear_size: int = 300,
    last_signal_size: int = 2042,
    dropout: int = 0.5,
    learning_rate: int = 1e-5):

        super().__init__()
        self.encoder = Encoder(in_channels_number, last_channel_size, out_dim, act_fun, conv1_size, conv2_size, conv3_size, kernel_size, pool_size, linear_size, last_signal_size, dropout)
        self.decoder = Decoder(in_channels_number, last_channel_size, out_dim, act_fun, conv1_size, conv2_size, conv3_size, kernel_size, pool_size, linear_size, last_signal_size)


    def forward(self, x, last_channel_size: int = 64, last_signal_size: int = 2042):
        encoded, indices = self.encoder(x)
        decoded = self.decoder(encoded,indices,last_channel_size,last_signal_size)
        return decoded, encoded
		

    def configure_optimizers(self, learning_rate: int = 1e-5):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer


    def _get_reconstruction_loss(self, batch, last_channel_size: int = 64, last_signal_size: int = 2042):
        x, y = batch
        x_reconstructed, _ = self.forward(x,last_channel_size, last_signal_size)
        loss = F.mse_loss(x, x_reconstructed, reduction="mean")
        return loss


    def training_step(self, train_batch, batch_idx, last_channel_size: int = 64, last_signal_size: int = 2042):
        loss = self._get_reconstruction_loss(train_batch, last_channel_size, last_signal_size)                             
        self.log('train_loss', loss)
        return loss


    def validation_step(self, val_batch, batch_idx, last_channel_size: int = 64, last_signal_size: int = 2042):
        loss = self._get_reconstruction_loss(val_batch, last_channel_size, last_signal_size)                             
        self.log('val_loss', loss)
        return loss


        
