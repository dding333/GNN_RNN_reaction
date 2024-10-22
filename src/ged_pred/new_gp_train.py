import torch
import pickle
from model import GNN
from dgl.dataloading import GraphDataLoader
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train(args, data):
    path = '../saved/' + args.pretrained_model + '/'
    print('loading hyperparameters of pretrained model from ' + path + 'hparams.pkl')
    with open(path + 'hparams.pkl', 'rb') as f:
        hparams = pickle.load(f)
    with open(path + 'feature_enc.pkl', 'rb') as f:
        feature_encoder = pickle.load(f)
    print("..........here are the hparams..........")
    print(hparams)
    print("..........here is the feature_encoder..........")
    print(feature_encoder)
    breakpoint()
    

    print('loading pretrained model from ' + path + 'model.pt')
    # Add the RNN-related arguments to the GNN model initialization
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # Replace args with hparams when initializing the GNN model
    model = GNN(hparams['gnn'], hparams['layer'], hparams['feature_len'], hparams['dim'],
                rnn_input_dim=hparams['rnn_input_dim'], rnn_hidden_size=hparams['rnn_hidden_size'], 
                latent_size=hparams['rnn_latent_size'], device=device)
    
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(path + 'model.pt', map_location=torch.device('cuda:0')))
        model = model.cuda(args.gpu)
    else:
        model.load_state_dict(torch.load(path + 'model.pt', map_location=torch.device('cpu')))


    dataloader = GraphDataLoader(data, batch_size=args.batch_size, shuffle=True)
    all_features1 = []
    all_features2 = []
    all_labels = []
    with torch.no_grad():
        model.eval()
        for graphs1, graphs2, labels, smiles1, smiles2  in dataloader:
            # Convert SMILES strings to one-hot encoded embeddings
            smiles1_embedding = smiles_to_onehot(smiles1, feature_encoder, hparams['rnn_input_dim']).to(device)
            smiles2_embedding = smiles_to_onehot(smiles2, feature_encoder, hparams['rnn_input_dim']).to(device)

            # Get graph and sequence embeddings
            graph_embeddings1 = model(graphs1, smiles1_embedding)
            graph_embeddings2 = model(graphs2, smiles2_embedding)

            all_features1.append(graph_embeddings1)
            all_features2.append(graph_embeddings2)
            all_labels.append(labels)
        
        all_features1 = torch.cat(all_features1, dim=0)
        all_features2 = torch.cat(all_features2, dim=0)

        # Combine or subtract features depending on the mode
        if args.feature_mode == 'concat':
            all_features = torch.cat([all_features1, all_features2], dim=-1).cpu().numpy()
        elif args.feature_mode == 'subtract':
            all_features = (all_features1 - all_features2).cpu().numpy()

        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

    print('splitting dataset')
    train_features = all_features[: int(0.8 * len(data))]
    train_labels = all_labels[: int(0.8 * len(data))]
    valid_features = all_features[int(0.8 * len(data)): int(0.9 * len(data))]
    valid_labels = all_labels[int(0.8 * len(data)): int(0.9 * len(data))]
    test_features = all_features[int(0.9 * len(data)):]
    test_labels = all_labels[int(0.9 * len(data)):]

    print('training the regression model\n')
    pred_model = SVR()
    pred_model.fit(train_features, train_labels)
    run_regression(pred_model, 'train', train_features, train_labels)
    run_regression(pred_model, 'valid', valid_features, valid_labels)
    run_regression(pred_model, 'test', test_features, test_labels)


def run_regression(model, mode, features, labels):
    pred = model.predict(features)
    mae = mean_absolute_error(labels, pred)
    rmse = mean_squared_error(labels, pred, squared=False)
    print('%s mae: %.4f   rmse: %.4f' % (mode, mae, rmse))


def smiles_to_onehot(smiles_list, feature_encoder, embedding_dim, max_smiles_length=None):
    """Convert SMILES strings to one-hot encoded embeddings with padding."""
    onehot_list = []
    
    if max_smiles_length is None:
        max_smiles_length = max(len(smiles) for smiles in smiles_list)
    
    for smiles in smiles_list:
        onehot_tensor = torch.zeros(max_smiles_length, embedding_dim)
        for idx, char in enumerate(smiles):
            if idx >= max_smiles_length:
                break
            # Assuming 'feature_encoder' is an accessible global variable or passed to this function
            if char in feature_encoder['element']:
                onehot_tensor[idx, feature_encoder['element'][char]] = 1
            else:
                onehot_tensor[idx, feature_encoder['element']['unknown']] = 1
        onehot_list.append(onehot_tensor)

    return torch.stack(onehot_list)
