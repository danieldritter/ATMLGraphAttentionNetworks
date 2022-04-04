import pickle
import torch
from utils import gen_graph
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, GNNBenchmarkDataset
from torch_geometric.nn import GATConv, GCNConv, GINConv
from torch_geometric.loader import DataLoader
from GAT import GraphAttentionLayer
from GATNet import GATNet
from sklearn.metrics import f1_score
from tqdm import tqdm 

# Hyper-Parameters

LEARNING_RATE = 0.005
WEIGHT_DECAY = .0005
HIDDEN_FEATURES = 8
CUR_MODEL = 'GAT' # Options: GAT, GATGeometric, GCN

USE_EARLY_STOPPING = True
FORCED_EPOCHS = 2
EARLY_STOPPING_PATIENCE = 100
NUM_EPOCHS = 10000
LOGGING_FREQUENCY = 2
NUM_RUNS = 5
BATCH_SIZE = 16
VERBOSE = True
NUM_RUNS = 5


if __name__ == "__main__":

    total_avg = 0.0
    total_avg_list = []
    datasetTrain = GNNBenchmarkDataset("./data","PATTERN","train")
    datasetVal = GNNBenchmarkDataset("./data","PATTERN","val")
    datasetTest = GNNBenchmarkDataset("./data","PATTERN","test")
    train_loader = DataLoader(datasetTrain, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(datasetVal, batch_size=BATCH_SIZE)
    test_loader = DataLoader(datasetTest, batch_size=BATCH_SIZE)
    num_features = 3
    num_classes = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for i in range(NUM_RUNS):
        val_losses = [] 
        val_accs = []
        if VERBOSE:
            print('Starting run number: ' + str(i + 1))
        model = GATNet(CUR_MODEL, "PATTERN", num_features).to(device)
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
                loss = F.nll_loss(out, batch.y)
                loss.backward()
                optimizer.step()
            model.eval()                
            if USE_EARLY_STOPPING:
                if epoch >= FORCED_EPOCHS - 1:
                    losses = 0.0
                    accs = 0.0
                    preds = []
                    labels = []
                    for batch in val_loader:
                        batch = batch.to(device)
                        out = model(batch)
                        pred = out.argmax(dim=1)
                        loss = F.nll_loss(out, batch.y)
                        losses += loss.item()
                        accs += torch.sum(pred == batch.y)/batch.y.shape[0]
        
                    avg_acc = accs/len(val_loader)
                    avg_loss = losses/len(val_loader)
                    val_losses.append(avg_loss)
                    val_accs.append(avg_acc)
                    if avg_acc >= cur_max or avg_loss <= cur_min_loss:
                        if VERBOSE:
                            print('Found new validation maximum at epoch ' + str(epoch + 1) + '!')
                            print('    Old max acc: ' + str(cur_max) + '%')
                            print('    New max acc: ' + str(avg_acc) + '%')
                            print('    Old max loss: ' + str(cur_min_loss) + '%')
                            print('    New max loss: ' + str(avg_loss) + '%')
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

                        val_accs.append(accs/len(val_loader))
                        val_losses.append(losses/len(val_loader))
                        print('Epoch: ' + str(epoch + 1) + ', Validation Accuracy: ' + str(acc) + '%')
                if epoch >= NUM_EPOCHS - 1:
                    stop_training = True
            epoch = epoch + 1
        gen_graph(val_accs, "validation_accuracy", i)
        gen_graph(val_losses, "validation_losses", i)
        model.eval()
        if USE_EARLY_STOPPING:
            model.load_state_dict(torch.load("./model/cur_model.pt"))
        accs = 0.0 
        for batch in test_loader:
            batch = batch.to(device)
            pred = out.argmax(dim=1)
            acc = torch.sum(pred == batch.y)/batch.y.shape[0]
            accs += acc

        print(f'Test Accuracy: {accs/len(test_loader):.4f}%')
        total_avg += accs/len(test_loader)
        total_avg_list.append(accs/len(test_loader))

    print('All Results: ' + str(total_avg_list))
    print(f'Total Test Average: {total_avg/NUM_RUNS}')
