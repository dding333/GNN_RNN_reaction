import os
import dgl
import torch
import pickle
import pysmiles
from collections import defaultdict

attribute_names = ['element', 'charge', 'aromatic', 'hcount']


class SmilesDataset(dgl.data.DGLDataset):
    def __init__(self, args, mode, feature_encoder=None, raw_graphs=None, sequence_data=None):
        self.args = args
        self.mode = mode
        self.feature_encoder = feature_encoder
        self.raw_graphs = raw_graphs
        self.sequence_data = sequence_data  # Add sequence data (e.g., SMILES strings)
        self.path = '../data/' + self.args.dataset + '/cache/' + self.mode
        self.reactant_graphs = []
        self.product_graphs = []
        super().__init__(name='Smiles_' + mode)

    def to_gpu(self):
        if torch.cuda.is_available():
            print('moving ' + self.mode + ' data to GPU')
            self.reactant_graphs = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.reactant_graphs]
            self.product_graphs = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.product_graphs]

    def save(self):
        print('saving ' + self.mode + ' reactants to ' + self.path + '_reactant_graphs.bin')
        print('saving ' + self.mode + ' products to ' + self.path + '_product_graphs.bin')
        dgl.save_graphs(self.path + '_reactant_graphs.bin', self.reactant_graphs)
        dgl.save_graphs(self.path + '_product_graphs.bin', self.product_graphs)

        # Save the sequence data as well
        with open(self.path + '_sequence_data.pkl', 'wb') as f:
            pickle.dump(self.sequence_data, f)

    def load(self):
        print('loading ' + self.mode + ' reactants from ' + self.path + '_reactant_graphs.bin')
        print('loading ' + self.mode + ' products from ' + self.path + '_product_graphs.bin')
        self.reactant_graphs = dgl.load_graphs(self.path + '_reactant_graphs.bin')[0]
        self.product_graphs = dgl.load_graphs(self.path + '_product_graphs.bin')[0]

        # Load the sequence data
        with open(self.path + '_sequence_data.pkl', 'rb') as f:
            self.sequence_data = pickle.load(f)

        self.to_gpu()

    def process(self):
        print('transforming ' + self.mode + ' data from networkx graphs to DGL graphs')
        for i, (raw_reactant_graph, raw_product_graph) in enumerate(self.raw_graphs):
            if i % 10000 == 0:
                print('%dk' % (i // 1000))
            reactant_graph = networkx_to_dgl(raw_reactant_graph, self.feature_encoder)
            product_graph = networkx_to_dgl(raw_product_graph, self.feature_encoder)
            self.reactant_graphs.append(reactant_graph)
            self.product_graphs.append(product_graph)
        self.to_gpu()

    def has_cache(self):
        return os.path.exists(self.path + '_reactant_graphs.bin') and os.path.exists(self.path + '_product_graphs.bin')

    def __getitem__(self, i):
        return self.reactant_graphs[i], self.product_graphs[i], self.sequence_data[i]  # Include sequence data

    def __len__(self):
        return len(self.reactant_graphs)


def networkx_to_dgl(raw_graph, feature_encoder):
    src = [s for (s, _) in raw_graph.edges]
    dst = [t for (_, t) in raw_graph.edges]
    graph = dgl.graph((src, dst), num_nodes=len(raw_graph.nodes))
    node_features = []
    for i in range(len(raw_graph.nodes)):
        raw_feature = raw_graph.nodes[i]
        numerical_feature = []
        for j in attribute_names:
            if raw_feature[j] in feature_encoder[j]:
                numerical_feature.append(feature_encoder[j][raw_feature[j]])
            else:
                numerical_feature.append(feature_encoder[j]['unknown'])
        node_features.append(numerical_feature)
    node_features = torch.tensor(node_features)
    graph.ndata['feature'] = node_features
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)
    return graph


def read_data(dataset, mode):
    path = '../data/' + dataset + '/' + mode + '.csv'
    print('preprocessing %s data from %s' % (mode, path))

    all_values = defaultdict(set)
    graphs = []
    sequence_data = []  # Store sequence data

    with open(path) as f:
        for line in f.readlines():
            idx, product_smiles, reactant_smiles, _ = line.strip().split(',')

            if len(idx) == 0:
                continue

            if int(idx) % 10000 == 0:
                print('%dk' % (int(idx) // 1000))

            reactant_graph = pysmiles.read_smiles(reactant_smiles, zero_order_bonds=False)
            product_graph = pysmiles.read_smiles(product_smiles, zero_order_bonds=False)

            if mode == 'train':
                for graph in [reactant_graph, product_graph]:
                    for attr in attribute_names:
                        for _, value in graph.nodes(data=attr):
                            all_values[attr].add(value)

            graphs.append([reactant_graph, product_graph])
            sequence_data.append((reactant_smiles, product_smiles))  # Store SMILES strings

    if mode == 'train':
        return all_values, graphs, sequence_data
    else:
        return graphs, sequence_data


def get_feature_encoder(all_values):
    feature_encoder = {}
    idx = 0
    for key, values in all_values.items():
        feature_encoder[key] = {}
        for value in values:
            feature_encoder[key][value] = idx
            idx += 1
        feature_encoder[key]['unknown'] = idx
        idx += 1

    return feature_encoder


def preprocess(dataset):
    print('preprocessing %s dataset' % dataset)

    all_values, train_graphs, train_sequences = read_data(dataset, 'train')
    valid_graphs, valid_sequences = read_data(dataset, 'valid')
    test_graphs, test_sequences = read_data(dataset, 'test')

    feature_encoder = get_feature_encoder(all_values)

    path = '../data/' + dataset + '/cache/feature_encoder.pkl'
    print('saving feature encoder to %s' % path)
    with open(path, 'wb') as f:
        pickle.dump(feature_encoder, f)

    return feature_encoder, (train_graphs, train_sequences), (valid_graphs, valid_sequences), (test_graphs, test_sequences)


def load_data1(args):
    cache_path = '../data/' + args.dataset + '/cache/'
    feature_encoder_path = cache_path + 'feature_encoder.pkl'

    # Check for preprocessed cache
    print(f"Checking if cache exists: {os.path.exists(cache_path)}")
    print(f"Checking if feature encoder exists: {os.path.exists(feature_encoder_path)}")

    if not args.force_preprocess:
        print("in if")
        print("Cache found, loading data.")
        
        # Load feature encoder from cache
        with open(feature_encoder_path, 'rb') as f:
            feature_encoder = pickle.load(f)

        # Load cached datasets (DGL graphs and sequence data)
        train_dataset = SmilesDataset(args, 'train')
        valid_dataset = SmilesDataset(args, 'valid')
        test_dataset = SmilesDataset(args, 'test')

        # Load DGL graphs and sequence data from disk
        train_dataset.load()
        valid_dataset.load()
        test_dataset.load()

        # Here we pass the `feature_encoder`, loaded DGL graphs, and sequence data
        train_dataset = SmilesDataset(args, 'train', feature_encoder, train_dataset.reactant_graphs, train_dataset.sequence_data)
        valid_dataset = SmilesDataset(args, 'valid', feature_encoder, valid_dataset.reactant_graphs, valid_dataset.sequence_data)
        test_dataset = SmilesDataset(args, 'test', feature_encoder, test_dataset.reactant_graphs, test_dataset.sequence_data)

    else:
        print("in else")
        print('Cache not found or force_preprocess is set to True. Preprocessing data...')
        os.makedirs(cache_path, exist_ok=True)

        # Preprocess data and store it
        feature_encoder, train_data, valid_data, test_data = preprocess(args.dataset)

        # Create datasets with the preprocessed data
        train_dataset = SmilesDataset(args, 'train', feature_encoder, *train_data)
        valid_dataset = SmilesDataset(args, 'valid', feature_encoder, *valid_data)
        test_dataset = SmilesDataset(args, 'test', feature_encoder, *test_data)

        # Save datasets after preprocessing (DGL and sequence data)
        train_dataset.save()
        valid_dataset.save()
        test_dataset.save()

    return feature_encoder, train_dataset, valid_dataset, test_dataset

def load_data2(args):
    cache_path = '../data/' + args.dataset + '/cache/'
    feature_encoder_path = cache_path + 'feature_encoder.pkl'

    # Check for preprocessed cache
    print(f"Checking if cache exists: {os.path.exists(cache_path)}")
    print(f"Checking if feature encoder exists: {os.path.exists(feature_encoder_path)}")

    if not args.force_preprocess:
        print("Cache found, loading data.")
        
        # Load feature encoder from cache
        with open(feature_encoder_path, 'rb') as f:
            feature_encoder = pickle.load(f)

        # Load cached datasets (DGL graphs and sequence data)
        train_dataset = SmilesDataset(args, 'train', feature_encoder)
        valid_dataset = SmilesDataset(args, 'valid', feature_encoder)
        test_dataset = SmilesDataset(args, 'test', feature_encoder)

        # Load DGL graphs and sequence data from disk (graphs and sequences are stored together)
        #train_dataset.load()  # `self.sequence_data` and graphs are loaded
        #valid_dataset.load()
        #test_dataset.load()

        # Directly assign the loaded data (avoiding redundant SmilesDataset creation)
        #train_dataset.feature_encoder = feature_encoder
        #valid_dataset.feature_encoder = feature_encoder
        #test_dataset.feature_encoder = feature_encoder

    else:
        print("Cache not found or force_preprocess is set to True. Preprocessing data...")
        os.makedirs(cache_path, exist_ok=True)

        # Preprocess data and store it
        feature_encoder, train_data, valid_data, test_data = preprocess(args.dataset)

        # Create datasets with the preprocessed data
        train_dataset = SmilesDataset(args, 'train', feature_encoder, *train_data)
        valid_dataset = SmilesDataset(args, 'valid', feature_encoder, *valid_data)
        test_dataset = SmilesDataset(args, 'test', feature_encoder, *test_data)

        # Save datasets after preprocessing (DGL and sequence data)
        #train_dataset.save()
        #valid_dataset.save()
        #test_dataset.save()

    return feature_encoder, train_dataset, valid_dataset, test_dataset

