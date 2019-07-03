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

    total_loss = 0
	for batch in tqdm.tqdm(train_iterator, total=params.train_size//params.batch_size):

		# compute model output and loss
		output = model(batch.input)
		loss = loss_fn(output, batch.target)
		total_loss = total_loss + loss

		# clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

    logging.info("- Train entropy loss: " + total_loss)


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


	best_val_loss = torch.tensor(float("Inf"))

	# training on num_epochs
	for epoch in range(params.num_epochs):
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # Run one epoch
        train(model, optimizer, loss_fn, train_iterator, params)
        val_loss = evaluate(model, optimizer, loss_fn, val_iterator, params)

        # Save best model
        if val_loss <= best_val_loss:
        	best_val_loss = val_loss

        	path = os.path.join(args.experiment_dir, 'best.pth.tar')
        	torch.save({'epoch': epoch+1,
        				'model': model.state_dict(),
        				'optimizer': optim.state_dict,
        				'loss': best_val_loss},
        				path)

        # Save latest model
        path = os.path.join(args.experiment_dir, 'latest.pth.tar')
    	torch.save({'epoch': epoch+1,
    				'model': model.state_dict(),
    				'optimizer': optim.state_dict,
    				'loss': val_loss},
    				path)


if __name__ == '__main__':
	args = parser.parse_args()

	# Load parameters from json file
	json_path = os.path.join(args.experiment_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    
    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.experiment_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    
    # Load data and get iterator
    data_handler = DataHandler(dir_path=args.data_dir,
    						   filenames={'train': 'train.csv',
    						   			  'validation': 'val.csv'.
    						   			  'test': None})
    data_handler.load_vocab('./model/TEXT.Field')

    train_iter, val_iter = data_handler.gen_iterator(batch_size=params.batch_size)
    train_size, val_size = data_handler.data_size

    params.train_size = train_size
    params.val_size   = val_size

    logging.info('- done.')

    # Define model
    model = net.Net(params.embedding_dim,
    				params.lstm_hidden_dim,
    				params.fc_hidden_dim,
    				params.vocab_size,
    				params.num_layers,
    				params.bidirectional)
    if params.cuda: model.cuda()

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, optimizer, loss_fn, train_iter, val_iter, params)