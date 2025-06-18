import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Wczytanie danych treningowych
with open('dane_wolumen.pkl', 'rb') as file:
    train_data_volume_raw = pickle.load(file)

train_data_volume = train_data_volume_raw.iloc[1:]

with open('dane_zmiana_ceny.pkl', 'rb') as file:
    train_data_price = pickle.load(file)

  # Wczytanie danych testowych
with open('dane_wolumen_test.pkl', 'rb') as file:
    test_data_volume_raw = pickle.load(file)

test_data_volume = test_data_volume_raw.iloc[1:]

with open('dane_zmiana_ceny_test.pkl', 'rb') as file:
    test_data_price = pickle.load(file)

# Proste przekształcenia danych (sprowadzenie zmian cen do logarytmów a wolumenu bezwzględnego do względnego)
train_data_price_log = np.log(train_data_price/100 + 1) *100
test_data_price_log = np.log(test_data_price/100 + 1) *100

train_data_volume_diff = train_data_volume_raw.diff()
train_data_volume_diff = train_data_volume_diff.iloc[1:]

test_data_volume_diff = test_data_volume_raw.diff()
test_data_volume_diff = test_data_volume_diff.iloc[1:]

datasets = [train_data_volume_diff, train_data_price_log, test_data_volume_diff, test_data_price_log]
output_datasets = []

# Wyrzucenie dwóch spółek, dla których występują braki danych
for dataset in datasets:
  dataset = dataset.drop(columns=['EKP.WA', 'MOL.WA'])
  output_datasets.append(dataset)

train_data_volume_clear = output_datasets[0]
train_data_price_clear = output_datasets[1]
test_data_volume_clear = output_datasets[2]
test_data_price_clear = output_datasets[3]

trainset_raw = pd.merge(train_data_volume_clear, train_data_price_clear, right_index=True, left_index=True)
testset_raw = pd.merge(test_data_volume_clear, test_data_price_clear, right_index=True, left_index=True)

print(trainset_raw.shape)
print(testset_raw.shape)

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize
from sklearn.preprocessing import StandardScaler

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Standaryzacja danych
scaler = StandardScaler().fit(trainset_raw)
trainset = scaler.transform(trainset_raw)
trainset = torch.tensor(trainset, dtype=torch.float32)
testset = scaler.transform(testset_raw)
testset = torch.tensor(testset, dtype=torch.float32)

print('Wektor treningowy:', trainset.size())
print('Wektor testowy:', testset.size())

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
  

train_dataset = MyDataset(trainset)
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)

test_dataset = MyDataset(testset)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

class AutoencoderModel(torch.nn.Module):
    def __init__(self, input_size=440, latent_dim=8):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, latent_dim)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, input_size),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

class EarlyStopping:
    def __init__(self, patience=15, delta=0.001, path='checkpoint.pt', verbose=True):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0

    def check_early_stop(self, train_loss):
        if self.best_loss is None or train_loss < self.best_loss - self.delta:
            self.best_loss = train_loss
            self.no_improvement_count = 0
            torch.save(model.state_dict(), self.path)
            if self.verbose:
                print(f"Model improved; checkpoint saved at loss {train_loss:.4f}")
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                print("Early stopping triggered.")
                return True
        return False
    
model = AutoencoderModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
print(model)

# Zapis modelu do pliku
import os
from datetime import datetime

date_now = datetime.strftime(datetime.now(), '%Y%m%d')

directory = 'Modele'
if not os.path.exists(directory):
  os.mkdir(directory)

save_path = os.path.join(directory, f'/autoencoder_{date_now}')
torch.save(model.state_dict(), save_path)

## TRENING I EWALUACJA
early_stopping = EarlyStopping()

loss_per_epoch = {'Epoch': [], 'Train_loss': []}
num_epochs = 100

for epoch in range(1, num_epochs+1):
    model.train()
    total_loss = 0.0

    for batch_data in train_dataloader:
        batch_data = batch_data.to(device)
        reconstructed = model(batch_data)
        loss = criterion(reconstructed, batch_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_data.size(0)

    avg_loss = total_loss / len(train_dataset)
    print(f"Epoch: {epoch} | train loss: {avg_loss}")
    loss_per_epoch['Epoch'].append(epoch)
    loss_per_epoch['Train_loss'].append(avg_loss)

    if early_stopping.check_early_stop(avg_loss):
        print(f'Stopping training at epoch: {epoch+1}')
        break
  
reconstruction_errors = []
model.eval()
with torch.no_grad():
    test_loss = 0.0
    for batch_data in test_dataloader:
        batch_data = batch_data.to(device)
        output = model(batch_data)

        batch_error = ((batch_data - output) ** 2).mean(dim=1)
        reconstruction_errors.append(batch_error.cpu())

    loss = criterion(output, batch_data)
    test_loss += loss.item() * batch_data.size(0)

test_loss /= len(test_dataset)

#reconstruction_errors = torch.cat(reconstruction_errors)
#reconstruction_errors

threshold = torch.quantile(reconstruction_errors, 0.98)
print(f'Próg błędu rekonstrukcji: {threshold:.6f}')
print('------------------------------------------------------------------------------------')

anomalies = np.where(reconstruction_errors > threshold)[0]
print('Liczba oznaczonych przypadków:', len(anomalies))

# Dla 0.98
anomaly_dates_098 = [testset_raw.index[i] for i in anomalies]
print(anomaly_dates_098)

### WYKRES LOSS VS EPOCHS
loss_per_epoch_df = pd.DataFrame.from_dict(loss_per_epoch)
plt.plot('Epoch', 'Train_loss', data=loss_per_epoch_df)
plt.xlabel('Liczba epok')
plt.ylabel('Strata')
plt.title('Wpływ hiperparametru learning_rate na stratę')
plt.grid(True)
plt.tight_layout()

save_path_lr = 'wplyw_lr_na_strate.png'
plt.savefig(save_path_lr)
plt.show()

## K-MEANS
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

latents = []

with torch.no_grad():
  for batch_data in train_dataloader:
    latent = model.encoder(batch_data.to(device))
    latents.append(latent.cpu())

latents_next = torch.cat(latents).numpy()
print(type(latent))

latent_std = StandardScaler().fit_transform(latents_next)

k_range = range(1, 21)
def calculate_kmeans_inertia(clusters, training_data):
    inertias = []
    for num_clusters in clusters:
        kmeans = KMeans(n_clusters=num_clusters, random_state=1274)
        kmeans.fit(training_data)
        inertias.append(kmeans.inertia_)

    return inertias

inertias = calculate_kmeans_inertia(k_range, latent_std)

plt.figure(figsize=(8, 6))
plt.plot(k_range, inertias)
plt.title('Wykres wartości inercji względem liczby klastrów')
plt.xlabel('Liczba klastrów')
plt.ylabel('Wartość inercji')
plt.xticks(k_range)
plt.tight_layout()

save_path_elbow = 'wykres_inercji.png'
plt.savefig(save_path_elbow)
plt.show()

k = 4
kmeans_model = KMeans(n_clusters=k, random_state=1274)
labels = kmeans_model.fit_predict(latent_std)

pca = PCA(n_components=2, random_state=1274)
latent_pca = pca.fit_transform(latent_std)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(latent_pca[:, 0], latent_pca[:, 1], c=labels,
                   s=8, cmap='tab10', alpha=0.8)

handles, texts = scatter.legend_elements(prop='colors')
new_labels = [f'Klaster {t}' for t in texts]

ax.legend(handles, new_labels, title='Klaster', loc='best', frameon=False)

ax.set_title('Rozmieszczenie klastrów w 2 wymiarowej przestrzeni')
ax.set_xlabel('PCA_1')
ax.set_ylabel('PCA_2')
plt.tight_layout()

save_path_clusters = 'klastry_kmeans_2d.png'
plt.savefig(save_path_clusters)
plt.show()

kmeans_silhouette = silhouette_score(latent_std, labels)
print(kmeans_silhouette)

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, init='pca', random_state=1274)
latent_tsne = tsne.fit_transform(latent_std)

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(latent_tsne[:, 0], latent_tsne[:, 1], latent_tsne[:, 2],
                     c=labels, s=10)
ax.set_title(f't-SNE 3D – k-means (k={len(np.unique(labels))})')
ax.set_xlabel('t-SNE-1')
ax.set_ylabel('t-SNE-2')
ax.set_zlabel('t-SNE-3')
plt.tight_layout()

save_path_tsne = 'klastry_kmeans_3d.png'
plt.savefig(save_path_tsne)
plt.show()

## Wyznaczenie dni do rewizji 
latents_test = []

with torch.no_grad():
  for batch_data in test_dataloader:
    latent_test = model.encoder(batch_data.to(device))
    latents_test.append(latent_test.cpu())

latents_test = torch.cat(latents_test).numpy()
print(type(latents_test))

latents_test_std = scaler.transform(latents_test)

# Predykcja
labels_test = kmeans_model.predict(latents_test_std)
print(np.unique(labels_test))

# Znalezienie odstających punktów
distances = np.linalg.norm(latents_test_std - kmeans_model.cluster_centers_[labels_test], axis=1)
print(distances)

kmeans_thresholds = {}
for cluster in list(np.unique(labels_test)):
  print(cluster)
  thr = np.percentile(distances[labels_test == cluster], 99)
  kmeans_thresholds[cluster] = float(thr)

outliers = np.array([distance > kmeans_thresholds[cluster] for distance, cluster in zip(distances, labels_test)])

print(f"Zbiór testowy zawiera {outliers.sum()} punktów uznanych za odstające w zbiorze {len(outliers)} obserwacji!")

outlier_idx = np.where(outliers)[0]
print(outlier_idx)

kmeans_outliers_dates_098 = [testset_raw.index[i] for i in outlier_idx]
print(kmeans_outliers_dates_098)
