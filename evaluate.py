import argparse
import logging
import os
import torch
import model.net as net


parser = argparse.ArgumentParser()
#parser.add_argument('--')


def evaluate(model, optimizer, loss_fn, val_iterator, params):
    """Evaluate the model on evaluation data

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        val_iterator: (generator) a generator that generates batches of data and labels
        params: (Params) hyperparameters
    """

    with torch.no_grad():
        model.eval()

        total_loss = 0.
        total_perplexity = 0.
        num_steps = params.val_size//params.batch_size

        for batch in val_iterator:

            # compute model output, loss, and perplexity
            output = model(batch.input.to(params.device))
            loss = loss_fn(output, batch.target.to(params.device).long())
            perplexity = net.perplexity(output, batch.target.to(params.device).long())
        
            total_loss = total_loss + loss.item()
            total_perplexity = total_perplexity + perplexity.item()

        mean_loss = total_loss/num_steps
        mean_perplexity = total_perplexity/num_steps
        logging.info("- Evaluation metrics: {} ; {}".format(str(mean_loss), str(mean_perplexity)))

    return {'loss': mean_loss, 'perplexity':mean_perplexity}