import pickle
import torch
import numpy as np 
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, GNNBenchmarkDataset
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.loader import DataLoader
from GAT import GraphAttentionLayer
from GATNet import GATNet
from tqdm import tqdm 

# Hyper-Parameters

LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0
CUR_MODEL = 'GAT' # Options: GAT, GCN
CUR_DATASET = "PATTERN" # Options: PATTERN

USE_EARLY_STOPPING = True
FORCED_EPOCHS = 1
EARLY_STOPPING_PATIENCE = 1
NUM_EPOCHS = 1
LOGGING_FREQUENCY = 2
NUM_RUNS = 5
BATCH_SIZE = 16
VERBOSE = True

def accuracy_SBM(scores, targets):
    S = targets.cpu().numpy()
    C = np.argmax( torch.nn.Softmax(dim=1)(scores).cpu().detach().numpy() , axis=1 )
    CM = confusion_matrix(S,C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets==r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r,r]/ float(cluster.shape[0])
            if CM[r,r]>0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = 100.* np.sum(pr_classes)/ float(nb_classes)
    return acc

# TODO: Compute training weight and pass in as weight to nll_loss to solve balance issues. 

if __name__ == "__main__":
    total_avg = 0.0
    total_avg_list = []
    datasetTrain = GNNBenchmarkDataset("./data",CUR_DATASET,"train")
    datasetVal = GNNBenchmarkDataset("./data",CUR_DATASET,"val")
    datasetTest = GNNBenchmarkDataset("./data",CUR_DATASET,"test")
    train_loader = DataLoader(datasetTrain, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(datasetVal, batch_size=BATCH_SIZE)
    test_loader = DataLoader(datasetTest, batch_size=BATCH_SIZE)
    num_features = 3
    num_classes = 2
    labels = []
    pos = 0 
    total = 0
    for item in train_loader:
        labels.append(item.y)
        pos += torch.sum(item.y)
        total += item.y.shape[0]
    labels = torch.cat(labels).numpy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # weights = torch.tensor(compute_class_weight("balanced",classes=[0,1],y=labels)).to(device).float()

    # Their implementation 
    V = label.size(0)
    label_count = torch.bincount(label)
    label_count = label_count[label_count.nonzero()].squeeze()
    cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
    cluster_sizes[torch.unique(label)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes>0).float()

    pos_weight = (total-pos)/total
    neg_weight = (total - (1-pos))/total 
    weights = torch.tensor([neg_weight,pos_weight]).to(device)
    for i in range(NUM_RUNS):
        val_losses = [] 
        val_accs = []
        if VERBOSE:
            print('Starting run number: ' + str(i + 1))
        model = GATNet(CUR_MODEL, CUR_DATASET, num_features).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        if VERBOSE:
            print('Starting training...')

        epoch = 0
        stop_counter = 0
        cur_max = 0.0
        cur_min_loss = float("inf")
        stop_training = False
        while not stop_training:
            model.train()
            for batch in tqdm(train_loader):
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                loss = F.nll_loss(out, batch.y,weight=weights)
                loss.backward()
                optimizer.step()
            model.eval()                
            if USE_EARLY_STOPPING:
                if epoch >= FORCED_EPOCHS - 1:
                    losses = 0.0
                    accs = 0.0
                    total = 0
                    correct = 0
                    for batch in val_loader:
                        batch = batch.to(device)
                        out = model(batch)
                        pred = out.argmax(dim=1)
                        loss = F.nll_loss(out, batch.y,weight=weights)
                        losses += loss.item()
                        total += batch.y.shape[0]
                        correct += torch.sum(pred == batch.y) 
                        acc = accuracy_SBM(out,batch.y)
                        accs += acc 
                    print(accs/len(val_loader))       
                    avg_acc = correct/total
                    avg_loss = losses/len(val_loader)
                    val_losses.append(avg_loss)
                    val_accs.append(avg_acc)
                    if avg_acc > cur_max or avg_loss < cur_min_loss:
                        if VERBOSE:
                            print('Found new validation maximum at epoch ' + str(epoch + 1) + '!')
                            print('    Old max acc: ' + str(cur_max) + '%')
                            print('    New max acc: ' + str(avg_acc) + '%')
                            print('    Old min loss: ' + str(cur_min_loss) + '%')
                            print('    New min loss: ' + str(avg_loss) + '%')
                            print('')
                        if avg_acc >= cur_max and avg_loss <= cur_min_loss:
                            torch.save(model.state_dict(), "./model/cur_model.pt")
                        cur_max = max(avg_acc, cur_max)
                        cur_min_loss = min(cur_min_loss, avg_loss)
                        stop_counter = 0
                    else:
                        stop_counter = stop_counter + 1
                        if VERBOSE:
                            print('Did not do better at epoch ' + str(epoch + 1) + '.')
                            print('    Old max: ' + str(cur_max) + '%')
                            print('    Current score: ' + str(avg_acc) + '%')
                            print('')
                        if stop_counter >= EARLY_STOPPING_PATIENCE:
                            if VERBOSE:
                                print('Stopping training...')
                            stop_training = True
            else:
                if VERBOSE:
                    if not epoch == 0 and (epoch + 1) % LOGGING_FREQUENCY == 0:
                        losses = 0.0 
                        accs = 0.0 
                        for batch in val_loader:
                            batch = batch.to(device)
                            out = model(batch)
                            loss = F.nll_loss(out, batch.y)
                            losses += loss.item()
                            pred = out.argmax(dim=1)
                            acc = torch.sum(pred == batch.y)/batch.y.shape[0]
                            accs += acc

                        val_accs.append(accs.item()/len(val_loader))
                        val_losses.append(losses/len(val_loader))
                        print('Epoch: ' + str(epoch + 1) + ', Validation Accuracy: ' + str(acc) + '%')
                if epoch >= NUM_EPOCHS - 1:
                    stop_training = True
            epoch = epoch + 1
        model.eval()
        if USE_EARLY_STOPPING:
            model.load_state_dict(torch.load("./model/cur_model.pt"))
        accs = 0.0 
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            acc = torch.sum(pred == batch.y)/batch.y.shape[0]
            accs += acc
        
        print(f'Test Accuracy: {accs/len(test_loader):.4f}%')
        total_avg += accs/len(test_loader)
        total_avg_list.append(accs/len(test_loader))

    avg_acc = total_avg/NUM_RUNS
    stddev = np.stddev(total_avg_list)
    ci = 1.96*(stddev/np.sqrt(len(total_avg_list)))
    print('All Results: ' + str(total_avg_list))
    print(f'Total Test Average: {total_avg/NUM_RUNS} +/- {ci}')
