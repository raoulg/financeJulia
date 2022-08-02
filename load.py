from dgl.data.utils import load_graphs

def get_data(file = "tfinance"):
    graph, label_dict = load_graphs(file)
    graph = graph[0]
    graph.ndata['label'] = graph.ndata['label'].argmax(1)
    graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
    graph.ndata['feature'] = graph.ndata['feature'].float()
    return graph
