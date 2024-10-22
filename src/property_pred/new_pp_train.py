import torch
import pickle
from model import GNN
from dgl.dataloading import GraphDataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score



def train(args, data):
    path = '../saved/' + args.pretrained_model + '/'
    print('loading hyperparameters of pretrained model from ' + path + 'hparams.pkl')
    with open(path + 'hparams.pkl', 'rb') as f:
        hparams = pickle.load(f)
    with open(path + 'feature_enc.pkl', 'rb') as f:
        feature_encoder = pickle.load(f)
    #print("..........here are the hparams..........")
    #print(hparams)
    #print("..........here is the feature_encoder..........")
    #print(feature_encoder)
    

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
    all_features = []
    all_labels = []

    with torch.no_grad():
        model.eval()
        for graphs, labels, smiles_strings in dataloader:
            #print("here is the smiles stirng")
            #print(smiles_strings)
            #print("here is the label")
            #print(labels)
            # Process SMILES strings into one-hot encoded embeddings using RNN if provided
            reactant_smiles_embedding = smiles_to_onehot(smiles_strings, feature_encoder, hparams['rnn_input_dim'])
            reactant_smiles_embedding = reactant_smiles_embedding.to(device) if torch.cuda.is_available() else reactant_smiles_embedding
            
            # Pass both graphs and one-hot encoded SMILES to the model
            graph_embeddings = model(graphs, reactant_smiles_embedding)
            
            all_features.append(graph_embeddings)
            all_labels.append(labels)
            #print(all_labels)
            #breakpoint()

        all_features = torch.cat(all_features, dim=0).cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

    print('splitting dataset')
    train_features = all_features[: int(0.8 * len(data))]
    train_labels = all_labels[: int(0.8 * len(data))]
    valid_features = all_features[int(0.8 * len(data)): int(0.9 * len(data))]
    valid_labels = all_labels[int(0.8 * len(data)): int(0.9 * len(data))]
    test_features = all_features[int(0.9 * len(data)):]
    test_labels = all_labels[int(0.9 * len(data)):]

    print('training the classification model\n')
    pred_model = LogisticRegression(solver='liblinear')
    pred_model.fit(train_features, train_labels)
    run_classification(pred_model, 'train', train_features, train_labels)
    run_classification(pred_model, 'valid', valid_features, valid_labels)
    run_classification(pred_model, 'test', test_features, test_labels)


def run_classification(model, mode, features, labels):
    acc = model.score(features, labels)
    auc = roc_auc_score(labels, model.predict_proba(features)[:, 1])
    print('%s acc: %.4f   auc: %.4f' % (mode, acc, auc))

def smiles_to_onehot(smiles_list, feature_encoder, embedding_dim, max_smiles_length=None):
    """Convert SMILES strings to one-hot encoded embeddings with padding."""
    onehot_list = []
    
    
    # Determine max length (if not provided)
    if max_smiles_length is None:
        max_smiles_length = max(len(smiles) for smiles in smiles_list)
    
    for smiles in smiles_list:
        # Initialize the one-hot tensor with zeros and pad to max_smiles_length
        onehot_tensor = torch.zeros(max_smiles_length, embedding_dim)
        
        for idx, char in enumerate(smiles):
            if idx >= max_smiles_length:  # If SMILES is longer than the max length, truncate it
                break
            if char in feature_encoder['element']:
                onehot_tensor[idx, feature_encoder['element'][char]] = 1
            else:
                onehot_tensor[idx, feature_encoder['element']['unknown']] = 1
        
        onehot_list.append(onehot_tensor)

    return torch.stack(onehot_list)
