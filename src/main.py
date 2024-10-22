import os
import argparse
import data_processing
import train
from property_pred import new_pp_data_processing, new_pp_train
from ged_pred import new_gp_data_processing, new_gp_train
from visualization import visualize


def print_setting(args):
    print('\n===========================')
    for k, v, in args.__dict__.items():
        print('%s: %s' % (k, v))
    print('===========================\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the index of gpu device')
    
    '''
    # Pretraining / Chemical Reaction Prediction
    parser.add_argument('--task', type=str, default='pretrain', help='downstream task')
    parser.add_argument('--dataset', type=str, default='USPTO-479k', help='dataset name')
    parser.add_argument('--epoch', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--gnn', type=str, default='tag', help='name of the GNN model')
    parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
    parser.add_argument('--dim', type=int, default=512, help='dimension of molecule embeddings')
    parser.add_argument('--margin', type=float, default=4.0, help='margin in contrastive loss')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--save_model', type=bool, default=False, help='save the trained model to disk')

    # New RNN parameters
    parser.add_argument('--rnn_input_dim', type=int, default=256, help='input dimension for RNN')
    parser.add_argument('--rnn_hidden_size', type=int, default=512, help='hidden size for RNN')
    parser.add_argument('--rnn_latent_size', type=int, default=256, help='latent size for RNN')
    
    # Add force_preprocess argument
    parser.add_argument('--force_preprocess', action='store_true', help='force re-preprocessing of data')
    '''
    
    
    # molecule property prediction
    parser.add_argument('--task', type=str, default='property_pred', help='downstream task')
    parser.add_argument('--pretrained_model', type=str, default='tag_512', help='the pretrained model')
    parser.add_argument('--dataset', type=str, default='Tox21', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size for calling the pretrained model')
    

    '''
    # GED prediction
    parser.add_argument('--task', type=str, default='ged_pred', help='downstream task')
    parser.add_argument('--pretrained_model', type=str, default='sage_512', help='the pretrained model')
    parser.add_argument('--dataset', type=str, default='QM9', help='dataset name')
    parser.add_argument('--n_molecules', type=int, default=1000, help='the number of molecules to be sampled')
    parser.add_argument('--n_pairs', type=int, default=10000, help='the number of molecule pairs to be sampled')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size for calling the pretrained model')
    parser.add_argument('--feature_mode', type=str, default='concat', help='how to construct the input feature')
    '''

    '''
    # visualization
    parser.add_argument('--task', type=str, default='visualization', help='downstream task')
    parser.add_argument('--subtask', type=str, default='ring', help='subtask type: reaction, property, ged, size, ring')
    parser.add_argument('--pretrained_model', type=str, default='sage_512', help='the pretrained model')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size for calling the pretrained model')
    parser.add_argument('--dataset', type=str, default='BBBP', help='dataset name')
    '''

    args = parser.parse_args()
    args.task = 'property_pred'
    args.save_model = False
    print_setting(args)
    print('current working directory: ' + os.getcwd() + '\n')

    if args.task == 'pretrain':
        data = data_processing.load_data(args)
        train.train(args, data)
    elif args.task == 'property_pred':
        data = new_pp_data_processing.load_data(args)
        new_pp_train.train(args, data)
    elif args.task == 'ged_pred':
        data = new_gp_data_processing.load_data(args)
        new_gp_train.train(args, data)
    elif args.task == 'visualization':
        visualize.draw(args)
    else:
        raise ValueError('unknown task')




if __name__ == '__main__':
    main()
