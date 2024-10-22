import os
import time
import torch
import pickle
import data_processing
import numpy as np
from model import GNN
from copy import deepcopy
from dgl.dataloading import GraphDataLoader
import torch.nn.functional as F

def train(args, data):
    feature_encoder, train_data, valid_data, test_data = data
    feature_len = sum([len(feature_encoder[key]) for key in data_processing.attribute_names])

    # Add RNN-related arguments to the GNN model initialization
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    model = GNN(args.gnn, args.layer, feature_len, args.dim, rnn_input_dim=args.rnn_input_dim,
                rnn_hidden_size=args.rnn_hidden_size, latent_size=args.rnn_latent_size, device=device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_dataloader = GraphDataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if torch.cuda.is_available():
        model = model.cuda(args.gpu)

    best_model_params = None
    best_val_mrr = 0
    print('start training\n')

    print('initial case:')
    model.eval()
    evaluate(model, 'valid', valid_data, args, feature_encoder)  # Pass feature_encoder here
    print("evaluate ran")
    evaluate(model, 'test', test_data, args, feature_encoder)    # Pass feature_encoder here
    print()

    for i in range(args.epoch):
        print('epoch %d:' % i)
        start_time = time.time()

        # train
        model.train()
        for reactant_graphs, product_graphs, reactant_smiles, product_smiles in train_dataloader:
            #print("in the training loop")
            # Convert SMILES strings to one-hot encoded embeddings using the feature_encoder
            reactant_smiles_embedding = smiles_to_onehot(reactant_smiles, feature_encoder, args.rnn_input_dim).to(device)
            product_smiles_embedding = smiles_to_onehot(product_smiles, feature_encoder, args.rnn_input_dim).to(device)

            # Move reactant and product graphs to the same device
            reactant_graphs = reactant_graphs.to(device)
            product_graphs = product_graphs.to(device)

            # Pass graphs and embeddings to the model
            reactant_embeddings = model(reactant_graphs, reactant_smiles_embedding)
            product_embeddings = model(product_graphs, product_smiles_embedding)

            # Calculate loss and perform optimization
            loss = calculate_loss(reactant_embeddings, product_embeddings, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # End the timer for the epoch and calculate the elapsed time
        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Epoch {i} completed in {epoch_duration:.2f} seconds")

        # evaluate on the validation set
        val_mrr = evaluate(model, 'valid', valid_data, args, feature_encoder)  # Pass feature_encoder here
        evaluate(model, 'test', test_data, args, feature_encoder)              # Pass feature_encoder here

        # save the best model
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_model_params = deepcopy(model.state_dict())

        print()

    # evaluation on the test set
    print('final results on the test set:')
    model.load_state_dict(best_model_params)
    evaluate(model, 'test', test_data, args, feature_encoder)  # Pass feature_encoder here
    print()

    # save the model, hyperparameters, and feature encoder to disk
    if args.save_model:
        if not os.path.exists('../saved/'):
            print('creating directory: ../saved/')
            os.mkdir('../saved/')

        directory = '../saved/%s_%d' % (args.gnn, args.dim)
        if not os.path.exists(directory):
            os.mkdir(directory)

        print('saving the model to directory: %s' % directory)
        torch.save(best_model_params, directory + '/model.pt')
        with open(directory + '/hparams.pkl', 'wb') as f:
            hp_dict = {'gpu': args.gpu, 'task': args.task, 'dataset': args.dataset, 'epoch': args.epoch, 'batch_size': args.batch_size, 'gnn': args.gnn, 'layer': args.layer, 'dim': args.dim, 'margin': args.margin, 'lr': args.lr, 'save_model': args.save_model, 'rnn_input_dim': args.rnn_input_dim, 'rnn_hidden_size': args.rnn_hidden_size, 'rnn_latent_size': args.rnn_latent_size, 'force_preprocess': args.force_preprocess, 'feature_len': feature_len}
            pickle.dump(hp_dict, f)

        with open(directory + '/feature_enc.pkl', 'wb') as f:
            pickle.dump(feature_encoder, f)



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

def calculate_loss(reactant_embeddings, product_embeddings, args):
    dist = torch.cdist(reactant_embeddings, product_embeddings, p=2)
    pos = torch.diag(dist)
    mask = torch.eye(args.batch_size)
    if torch.cuda.is_available():
        mask = mask.cuda(args.gpu)
    neg = (1 - mask) * dist + mask * args.margin
    neg = torch.relu(args.margin - neg)
    loss = torch.mean(pos) + torch.sum(neg) / args.batch_size / (args.batch_size - 1)
    return loss
    
def evaluate(model, mode, data, args, feature_encoder):
    model.eval()
    with torch.no_grad():
        all_product_embeddings = []
        product_dataloader = GraphDataLoader(data, batch_size=args.batch_size)
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

        # Unpack reactant_graphs, product_graphs, reactant_smiles, product_smiles
        for _, product_graphs, _, product_smiles in product_dataloader:
            # Convert product_smiles to one-hot embeddings
            product_smiles_embedding = smiles_to_onehot(product_smiles, feature_encoder, args.rnn_input_dim).to(device)
            product_graphs = product_graphs.to(device)

            # Pass graphs and embeddings to the model
            product_embeddings = model(product_graphs, product_smiles_embedding)
            all_product_embeddings.append(product_embeddings)

        all_product_embeddings = torch.cat(all_product_embeddings, dim=0)

        # rank
        all_rankings = []
        reactant_dataloader = GraphDataLoader(data, batch_size=args.batch_size)
        i = 0
        for reactant_graphs, _, reactant_smiles, _ in reactant_dataloader:
            # Convert reactant_smiles to one-hot embeddings
            reactant_smiles_embedding = smiles_to_onehot(reactant_smiles, feature_encoder, args.rnn_input_dim).to(device)
            reactant_graphs = reactant_graphs.to(device)

            # Pass graphs and embeddings to the model
            reactant_embeddings = model(reactant_graphs, reactant_smiles_embedding)
            ground_truth = torch.unsqueeze(torch.arange(i, min(i + args.batch_size, len(data))), dim=1)
            i += args.batch_size
            if torch.cuda.is_available():
                ground_truth = ground_truth.cuda(args.gpu)

            dist = torch.cdist(reactant_embeddings, all_product_embeddings, p=2)
            sorted_indices = torch.argsort(dist, dim=1)
            rankings = ((sorted_indices == ground_truth).nonzero()[:, 1] + 1).tolist()
            all_rankings.extend(rankings)

        # calculate metrics
        all_rankings = np.array(all_rankings)
        mrr = float(np.mean(1 / all_rankings))
        mr = float(np.mean(all_rankings))
        h1 = float(np.mean(all_rankings <= 1))
        h3 = float(np.mean(all_rankings <= 3))
        h5 = float(np.mean(all_rankings <= 5))
        h10 = float(np.mean(all_rankings <= 10))

        print('%s  mrr: %.4f  mr: %.4f  h1: %.4f  h3: %.4f  h5: %.4f  h10: %.4f' % (mode, mrr, mr, h1, h3, h5, h10))
        return mrr

