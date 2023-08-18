import torch
import datetime
import tqdm

from torch.utils.tensorboard import SummaryWriter

def train_loop(training_loader, validation_loader, model, loss_fn, optimizer, epochs, locator: bool = False):
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_path = 'runs/run_{}'.format(timestamp)

    #TODO: Change tensorboard with online logger
    writer = SummaryWriter(run_path + 'risclip_train')
    epoch = 0

    best_vloss = 1_000_000.

    for epoch in tqdm(range(epochs)):
        print('EPOCH {}:'.format(epoch+1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch, training_loader, model, loss_fn, optimizer, locator, writer)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = calculateLoss(loss_fn, voutputs, vlabels, locator=locator)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch + 1)
        writer.flush()

        # Save last model weights
        torch.save(model.state_dict(), run_path + "weights/last.pt")

        # Save best model weights
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            torch.save(model.state_dict(), run_path + "weights/best.pt")

    

def train_one_epoch(epoch_index, training_loader, model, loss_fn, optimizer, locator, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)

        # If training Locator, use ground truth probability map as label
        # otherwise use bounding boxes
        loss = calculateLoss(loss_fn, outputs, labels, locator=locator)
        loss.backward()

        # Adjust learning weights
        # TODO: Train only desired layers
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def calculateLoss(loss_fn, output, labels, locator: bool = False):
    #FIXME: If ground truth pixel average is 0 (no segmentation map is found), skip this sample
    if locator:
        return loss_fn(output, labels['gt'])
    else:
        return loss_fn(output, labels['bbox'])