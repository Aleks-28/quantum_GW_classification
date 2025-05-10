import os
import time
import numpy as np
import pandas as pd
import warnings
from gwpy.timeseries import TimeSeries
from joblib import Parallel, delayed
import glob
from tqdm import tqdm
np.random.seed(42)


def create_dataframe(path,data_path,nbr_samples):
    training_labels = pd.read_csv(path)
    paths = glob.glob(data_path + "/*/*/*/*")
    ids = [path.split("/")[-1].split(".")[0] for path in paths]
    paths_df = pd.DataFrame({"path":paths, "id": ids})
    data = pd.merge(left=training_labels, right=paths_df, on="id")
    data.to_csv("./Utils/data.csv",index=True)
    data.head(5)
    df_reduced = data.sample(n=nbr_samples,random_state=42)
    df_reduced = df_reduced.reset_index(drop=True)
    return df_reduced


def read_file(filepath):
    data = np.load(filepath).squeeze()
    ts1 = TimeSeries(data[0,:], sample_rate=2048)
    ts2 = TimeSeries(data[1,:], sample_rate=2048)
    ts3 = TimeSeries(data[2,:], sample_rate=2048)
    return ts1, ts2, ts3

def preprocess(d1, d2, d3, bandpass=False, lf=35, hf=350):
    white_d1 = d1.whiten(window=("tukey",0.2))
    white_d2 = d2.whiten(window=("tukey",0.2))
    white_d3 = d3.whiten(window=("tukey",0.2))
    if bandpass: # bandpass filter
        bp_d1 = white_d1.bandpass(lf, hf) 
        bp_d2 = white_d2.bandpass(lf, hf)
        bp_d3 = white_d3.bandpass(lf, hf)      
        return bp_d1, bp_d2, bp_d3
    else: # only whiten
        return white_d1, white_d2, white_d3

def preprocessing_loop(row, df_reduced):
    # Preprocess data and save it into a new folder along with a new csv "df_preprocessed.csv" file with the new paths
    path=row.path
    folder_path='/'.join(path.split('/')[0:-1])
    folder_path_preprocessed = folder_path.replace("Original_Data", "Kaggle_GWData_preprocessed")
    os.makedirs(folder_path_preprocessed, exist_ok=True)
    s1,s2,s3=read_file(path)
    ps1,ps2,ps3=preprocess(s1, s2, s3, bandpass=True, lf=35, hf=350)
    np.save(folder_path_preprocessed+'/%s.npy'%row.id, [ps1,ps2,ps3])


def apply_preprocessing(df_reduced):
    Parallel(n_jobs=-1)(delayed(preprocessing_loop)(row, df_reduced) for row in tqdm(df_reduced.itertuples()))


def refactor(df_reduced):
    warnings.warn("deprecated", DeprecationWarning)
    df_preprocessed = df_reduced.replace({'Original_Data': 'Kaggle_GWData_preprocessed'}, regex=True)
    train_df = pd.DataFrame(columns=["id", "target", "path"])
    train_df["id"] = train_df["id"].astype(str)
    train_df["target"] = train_df["id"].astype(int)
    train_df["path"] = train_df["id"].astype(str)
    for row in tqdm(df_preprocessed.itertuples()):
        path=row.path
        folder_path='/'.join(path.split('/')[0:-1])
        folder_path_red_refact = folder_path.replace("Kaggle_GWData_preprocessed", "Kaggle_GWData_final")
        os.makedirs(folder_path_red_refact, exist_ok=True)
        data = np.load(path)
        d1,d2,d3 = data[0,:].squeeze(),data[1,:].squeeze(),data[2,:].squeeze()
        np.save(folder_path_red_refact+'/%s'%row.id + "_d1", d1)
        np.save(folder_path_red_refact+'/%s'%row.id + "_d2", d2)
        np.save(folder_path_red_refact+'/%s'%row.id + "_d3", d3)
        train_df = train_df.append(pd.Series({'id':"%s"%row.id + "_d1", 'target':row.target, 'path':folder_path_red_refact + '/%s'%row.id + "_d1.npy"}),ignore_index=True)
        train_df = train_df.append(pd.Series({'id':"%s"%row.id + "_d2", 'target':row.target, 'path':folder_path_red_refact + '/%s'%row.id + "_d2.npy"}),ignore_index=True)
        train_df = train_df.append(pd.Series({'id':"%s"%row.id + "_d3", 'target':row.target, 'path':folder_path_red_refact + '/%s'%row.id + "_d3.npy"}),ignore_index=True)
        train_df.to_csv("./Utils/train_df.csv")


if __name__ == "__main__":
    start=time.time()
    path_init = "./Utils/training_labels.csv"
    data_path = "/media/aleks/Games/KaggleData/Original_Data/train"
    nbr_samples = 10000
    df_reduced = create_dataframe(path_init,data_path,nbr_samples)
    apply_preprocessing(df_reduced)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refactor(df_reduced)
    stop=time.time()
    print('Elapsed time for the entire preprocessing: {:.2f} s'
          .format(stop - start))