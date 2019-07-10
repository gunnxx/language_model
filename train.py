import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

import utils
import model.net as net
from model.data_handler import DataHandler
from evaluate import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data', help='Directory containing datasets')
parser.add_argument('--experiment_dir', default='./experiments/base_model', help='Directory containing the experiment setup')
parser.add_argument('--restore_file', default=None, help='Training checkpoint file name inside experiment_dir (optional)')


def train(model, optimizer, loss_fn, train_iterator, params):
    """Train the model on one epoch

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        train_iterator: (generator) a generator that generates batches of data and labels
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()

    total_loss = 0.
    total_perplexity = 0.
    num_steps = params.train_size//params.batch_size

    for batch in tqdm.tqdm(train_iterator, total=num_steps):

        # compute model output, loss, and perplexity
        output = model(batch.input.to(params.device))
        loss = loss_fn(output, batch.target.to(params.device).long())
        perplexity = net.perplexity(output, batch.target.to(params.device).long())
        
        total_loss = total_loss + loss.item()
        total_perplexity = total_perplexity + perplexity.item()

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

    mean_loss = total_loss/num_steps
    mean_perplexity = total_perplexity/num_steps
    logging.info("- Training metrics: {} ; {}".format(str(mean_loss), str(mean_perplexity)))


def train_and_evaluate(model, optimizer, loss_fn, train_iterator, val_iterator, params):
    """Train the model and evaluate every epoch

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        train_iterator: (generator) a generator that generates batches of data and labels
        val_iterator: (generator) a generator that generates batches of data and labels
        params: (Params) hyperparameters
    """

    # reload weights from checkpoint_file if specified
    if args.restore_file:
        restore_path = os.path.join(args.experiment_dir, args.restore_file)

        logging.info("Restoring parameters from {}".format(restore_path))
        if not os.path.exists(restore_path):
            raise ("File doesn't exist")

        checkpoint = torch.load(restore_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    best_val_perplexity = 0.
    best_val_loss = float("Inf")

    # training on num_epochs
    for epoch in range(params.num_epochs):
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # Run one epoch
        train(model, optimizer, loss_fn, train_iterator, params)
        val_metric = evaluate(model, optimizer, loss_fn, val_iterator, params)

        # Save best model regards on loss
        if val_metric['loss'] <= best_val_loss:
            best_val_loss = val_metric['loss']

            path = os.path.join(args.experiment_dir, 'best_loss.pth.tar')
            torch.save({'epoch': epoch+1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': val_metric['loss'],
                        'perplexity': val_metric['perplexity']},
                        path)

        # Save best model regards on perplexity
        if val_metric['perplexity'] >= best_val_perplexity:
            best_val_perplexity = val_metric['perplexity']

            path = os.path.join(args.experiment_dir, 'best_perplexity.pth.tar')
            torch.save({'epoch': epoch+1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': val_metric['loss'],
                        'perplexity': val_metric['perplexity']},
                        path)

        # Save latest model
        path = os.path.join(args.experiment_dir, 'latest.pth.tar')
        torch.save({'epoch': epoch+1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': val_metric['loss'],
                    'perplexity': val_metric['perplexity']},
                    path)


if __name__ == '__main__':
    args = parser.parse_args()

    # Load parameters from json file
    json_path = os.path.join(args.experiment_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if torch.cuda.is_available(): torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.experiment_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    
    # Load data and get iterator
    vocab_path = './model/TEXT.Field'
    train_file_path = os.path.join(args.data_dir, 'train.csv')
    val_file_path = os.path.join(args.data_dir, 'val.csv')

    data_handler = DataHandler()
    data_handler.load_vocab(vocab_path)
    data_handler.load_dataset(train_file=train_file_path, val_file=val_file_path)

    train_iter, val_iter = data_handler.gen_iterator(batch_size=params.batch_size)
    train_size, val_size = data_handler.data_size

    params.train_size = train_size
    params.val_size   = val_size
    params.vocab_size = len(data_handler.vocab.itos)

    logging.info('- done.')

    # Define model
    model = net.Net(params.embedding_dim,
                    params.lstm_hidden_dim,
                    params.fc_hidden_dim,
                    params.vocab_size,
                    params.num_layers,
                    params.bidirectional)
    model.to(params.device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, optimizer, loss_fn, train_iter, val_iter, params)