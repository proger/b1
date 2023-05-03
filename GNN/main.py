import torch
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator #needed to extract and evaluate the ogb-ddi dataset
import matplotlib.pyplot as plt #needed to visualize loss curves
import numpy as np 

from models.GraphSAGE import GNNStack, LinkPredictor
from models.train import train
from models.test import test


def main():
    # Hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optim_wd = 0
    epochs = 300
    hidden_dim = 256
    dropout = 0.3
    num_layers = 2
    lr = 3e-3
    node_emb_dim = 256
    batch_size = 64 * 1024

    dataset = PygLinkPropPredDataset(name="ogbl-ddi", root='./dataset/') #download the dataset
    split_edge = dataset.get_edge_split()
    pos_train_edge = split_edge['train']['edge'].to(device)

    graph = dataset[0]
    edge_index = graph.edge_index.to(device)

    evaluator = Evaluator(name='ogbl-ddi')

    emb = torch.nn.Embedding(graph.num_nodes, node_emb_dim).to(device) # each node has an embedding that has to be learnt
    model = GNNStack(node_emb_dim, hidden_dim, hidden_dim, num_layers, dropout, emb=True).to(device) # the graph neural network that takes all the node embeddings as inputs to message pass and agregate
    link_predictor = LinkPredictor(hidden_dim, hidden_dim, 1, num_layers + 1, dropout).to(device) # the MLP that takes embeddings of a pair of nodes and predicts the existence of an edge between them

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(link_predictor.parameters()) + list(emb.parameters()),
        lr=lr, weight_decay=optim_wd
    )

    train_loss = []
    val_hits = []
    test_hits = []
    for e in range(epochs):
        loss = train(model, link_predictor, emb.weight, edge_index, pos_train_edge, batch_size, optimizer)
        print(f"Epoch {e + 1}: loss: {round(loss, 5)}")
        train_loss.append(loss)

        if (e+1)%10 ==0:
            result = test(model, link_predictor, emb.weight, edge_index, split_edge, batch_size, evaluator)
            val_hits.append(result['Hits@20'][0])
            test_hits.append(result['Hits@20'][1])
            print(result)

    plt.title('Link Prediction on OGB-ddi using GraphSAGE GNN')
    plt.plot(train_loss,label="training loss")
    plt.plot(np.arange(9,epochs,10),val_hits,label="Hits@20 on validation")
    plt.plot(np.arange(9,epochs,10),test_hits,label="Hits@20 on test")
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()


if __name__ == '__main__':
    main()