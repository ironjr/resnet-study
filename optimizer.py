import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train(model, optimizer, loader_train, loader_val=None,
          device=torch.device('cuda'), dtype_x=torch.float32,
          dtype_y = torch.long, num_epochs=1, logger=None, print_every=100,
          verbose=True):
    """Trains given model with given optimizer and data loader.

    Args:
        model (:obj:`torch.nn.Module`): A PyTorch Module for a model to be
            trained.
        optimizer (:obj:`torch.optim.optim`): A PyTorch Optimizer defining the
            training method.
        loader_train (:obj:`torch.utils.data.DataLoader`): DataLoader having
            training data.
        loader_val (:obj:`torch.utils.data.DataLoader`): DataLoader having
            validation data.
        device (:obj:`torch.device`, optional): Device where training is being
            held. Default is CUDA.
        dtype_x (:obj:`dtype`): Data type of input data. Default is
            torch.float32
        dtype_y (:obj:`dtype`): Data type of classifier. Default is torch.long
        num_epochs (int, optional): Number of epoches to be train.
        logger (:obj:`Logger`): Logs history for tensorboard statistics.
            Default is None.
        print_every (int, optional): Period of print of the statistics. Default
            is 100.
        verbose (bool, optional): Print the statistics in detail. Default is
            True.
    
    Returns: Nothing.
    """
    model = model.to(device=device)
    num_steps = len(loader_train)
    for e in range(num_epochs):
        for i, (x, y) in enumerate(loader_train):
            # Model to training mode
            model.train()

            x = x.to(device=device, dtype=dtype_x)
            y = y.to(device=device, dtype=dtype_y)

            # Forward path
            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Backward path
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute accuracy
            _, argmax = torch.max(scores, 1)
            accuracy = (y == argmax.squeeze()).float().mean()

            # Print the intermediate performance. Test for validation data if
            # it is given.
            if verbose and (i + 1) % print_every == 0:
                # Common statistics.
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.2f} %'
                    .format(e + 1, num_epochs, i + 1, num_steps, loss.item(),
                    accuracy.item() * 100), end='')
                
                # Validation dataset is provided.
                if loader_val is not None:
                    print(', ', end='')
                    test(model, loader_val, device=device, dtype_x=dtype_x,
                        dtype_y=dtype_y)
                else:
                    print('')
                
                # Tensorboard logging
                if logger is not None:
                    # 1. Scalar summary
                    info = { 'loss': loss.item(), 'accuracy': accuracy.item() }

                    for tag, value in info.items():
                        logger.log_scalar(tag, value, i + 1)
                    
                    # 2. Historgram summary
                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        logger.log_histogram(tag, value.data.cpu().numpy(), i + 1)
                        logger.log_histogram(tag + '/grad',
                            value.grad.data.cpu().numpy(), i + 1)
                        
                    # 3. Image summary
                    # info = { 'images': x.view(-1, 32, 32)[:10].cpu().numpy()}
                    # for tag, images in info.items():
                    #     logger.log_image(tag, images, i + 1)

def test(model, data_loader, device=torch.device('cuda'),
         dtype_x=torch.float32, dtype_y=torch.long):
    """
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            # Preprocessed image using FiveCrop or TenCrop
            # bs, ncrops, c, h, w = x.size()
            # y = y.to(device=device, dtype=dtype_y)
            # predicted_avg = 0
            # for i in range(ncrops):
            #     xx = x[:, i, :, :, :].view(-1, c, h, w)
            #     xx = xx.to(device=device, dtype=dtype_x)
            #     out = model(xx)
            #     _, predicted = torch.max(out.data, 1)
            #     predicted_avg += predicted
            # predicted_avg /= ncrops
            # total += y.size(0)
            # correct += (predicted_avg == y).sum().item()

            # For single image use below:
            x = x.to(device=device, dtype=dtype_x)
            y = y.to(device=device, dtype=dtype_y)
            out = model(x)
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        acc = float(correct) / total
        print('[{}/{}] Correct ({:.2f} %)'.format(correct, total, acc * 100))
