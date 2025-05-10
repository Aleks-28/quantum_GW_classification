import pandas as pd
import os
import time
from torch.utils.data import Dataset
from dataset import *
from src.model.autoencoder import *
from joblib import Parallel, delayed
from tqdm import tqdm
import mlflow.pytorch
import mlflow.pyfunc


def sample_data(df_path, num_samples, save_path):
    train_df = pd.read_csv(df_path,index_col=0)
    df = train_df.sample(num_samples,ignore_index=True,random_state=42)
    labels=df.loc[:,'target'].values
    print(labels)
    os.makedirs(save_path, exist_ok=True)
    np.save(save_path + "/labels.npy",labels)
    train_df.to_csv(save_path + "/train_df.csv")
    return df

def get_signals(dataset):
    return torch.stack([dataset[i][0] for i in range(len(dataset))], dim=0)

def features_extraction(input,index,save_path,logged_model):
    model = mlflow.pytorch.load_model(logged_model)
    os.makedirs(save_path, exist_ok=True)
    model.eval()
    with torch.no_grad():
        sig_pred = model(input.to(model.device))
    features = sig_pred[1].cpu().detach().squeeze().numpy()
    np.save(save_path+'/%s.npy'%index, features)

if __name__ == "__main__":
    save_path = "/media/aleks/Games/KaggleData/Kaggle_GWData_final/extracted_features"
    df_path = "./Utils/train_df.csv"
    logged_model = 'runs:/91ab0368798e433fb4407bd4df60d11d/model'
    df = sample_data(df_path,10000,save_path)
    dataset = TrainDataset(df)
    input = get_signals(dataset)
    start=time.time()
    Parallel(n_jobs=8)(delayed(features_extraction)(input[[i]],i,save_path,logged_model) for i in range(len(input)))
    stop=time.time()