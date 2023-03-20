from ogb.linkproppred import LinkPropPredDataset

dataset = LinkPropPredDataset(name='ogbl-ddi', root='data')
data = dataset[0]
print(data)