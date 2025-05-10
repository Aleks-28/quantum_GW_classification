from dataset import *
from model.autoencoder import *
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import MLFlowLogger
import pytorch_lightning as pl
import mlflow.pytorch


def main(save_path, train_data, val_data):
    data_loader = DataLoader(train_data, batch_size=60, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=60, shuffle=False, num_workers=8)
    env = mlflow.pytorch.get_default_conda_env()
    model = AutoEncoder(last_channel_size=64, out_dim = 16, act_fun=nn.ReLU)
    print(model)
    trainer = pl.Trainer(gpus=1, num_nodes=1, limit_train_batches=0.5, max_epochs = 30)
    mlflow.pytorch.autolog()
    trainer.fit(model, data_loader,val_loader)
    ## Saving the Model
    mlflow.pytorch.save_model(pytorch_model=model,path=save_path,conda_env=env)
    
if __name__ == "__main__":   
    train_df = pd.read_csv("./Utils/train_df.csv",index_col=0)
    dataset = TrainDataset(train_df)
    train_data, val_data = random_split(dataset, [22000, 8000])
    save_path = "/home/aleks/G2NKaggle_files/Model/MSE_Model_Save"
    main(save_path,train_data,val_data)