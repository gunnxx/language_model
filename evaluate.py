import argparse
import logging
import os


def evaluate(model, optimizer, loss_fn, val_iterator, params):
	"""Evaluate the model on evaluation data

    Args:
		model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        val_iterator: (generator) a generator that generates batches of data and labels
        params: (Params) hyperparameters
    """

    # set model to evaluation mode
    model.eval()

    total_loss = 0
    for batch in val_iterator:

    	# compute model output and loss
    	output = model(batch.input)
    	loss = loss_fn(output, batch.target)
    	total_loss = total_loss + loss

    logging.info("- Evaluation entropy loss: " + total_loss)
    return total_loss