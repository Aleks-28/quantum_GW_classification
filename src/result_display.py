from numpy import size
from torch.utils.data import Dataset
from dataset import *
from src.model.autoencoder import *
import matplotlib.pyplot as plt
from dataset import *
import feature_extraction
import mlflow.pytorch
import mlflow.pyfunc
from feature_extraction import get_signals



def visualize_reconstructions(model, input_sig):
    model.eval()
    with torch.no_grad():
        reconst_sig = model(input_sig.to(model.device))
    reconst_sig = reconst_sig[0].cpu().detach().squeeze().numpy()

    # Plotting
    plt.figure(figsize=(20,7))
    sig_original = input_sig.cpu().detach().squeeze().numpy()
    plt.plot(sig_original, c="blue", label="reconstructed")
    plt.plot(reconst_sig, c="yellow", label="original")
    plt.title("reconstructed from latent space = 16")
    plt.show()

if __name__ == "__main__":
    train_df = pd.read_csv("./Utils/train_df.csv",index_col=0)
    dataset = TrainDataset(train_df)
    print(train_df.path)
    cuda0 = torch.device('cuda:0')
    test_sig = get_signals(1,dataset)
    logged_model = '../mlruns/0/1649e22a27a2495ab1847d8cb20d89be/artifacts/model'
    model = mlflow.pytorch.load_model(logged_model)
    visualize_reconstructions(model,test_sig)


