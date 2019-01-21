import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train(model, optimizer, loader_train, loader_val=None,
          device=torch.device('cuda'), dtype_x=None,
          dtype_y=None, num_epochs=1, logger_train=None, logger_val=None,
          iteration_begins=0, print_every=100, verbose=True):
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
        dtype_x (:obj:`dtype`, optional): Data type of input data. Default is
            None.
        dtype_y (:obj:`dtype`, optional): Data type of classifier. Default is
            None.
        num_epochs (int, optional): Number of epoches to be train.
        logger_train (:obj:`Logger`, optional): Logs history for tensorboard
            statistics. Related to training set. Default is None.
        logger_val (:obj:`Logger`, optional): Logs history for tensorboard
            statistics. Related to validation set. Default is None.
        iteration_begins (int, optional): Tells the logger from where it counts
            the number of iterations passed. Default is 0.
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

            if dtype_x is not None:
                x = x.to(device=device, dtype=dtype_x)
            else:
                x = x.to(device=device)
            if dtype_y is not None:
                y = y.to(device=device, dtype=dtype_y)
            else:
                y = y.to(device=device)

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
                val_acc = None
                if loader_val is not None and len(loader_val) is not 0:
                    print(', ', end='')
                    val_acc = test(model, loader_val, device=device,
                        dtype_x=dtype_x, dtype_y=dtype_y)
                else:
                    print('')
                
                # Tensorboard logging training set statistics.
                iterations = e * num_steps + i + iteration_begins + 1
                if logger_train is not None:

                    # 1. Scalar summary
                    info = { 'loss': loss.item(), 'accuracy': accuracy.item() }
                    for tag, value in info.items():
                        logger_train.log_scalar(tag, value, iterations)
                    
                    # 2. Historgram summary
                    # for tag, value in model.named_parameters():
                    #     tag = tag.replace('.', '/')
                    #     logger_train.log_histogram(tag,
                    #         value.data.cpu().numpy(), iterations)
                    #     logger_train.log_histogram(tag + '/grad',
                    #         value.grad.data.cpu().numpy(), iterations)
                        
                    # 3. Image summary
                    # info = { 'images': x.view(-1, 32, 32)[:10].cpu().numpy()}
                    # for tag, images in info.items():
                    #     logger_train.log_image(tag, images, iterations)

                # Tensorboard logging validation set statistics.
                if val_acc is not None and logger_val is not None:

                    # 1. Scalar summary
                    info = { 'accuracy': val_acc }
                    for tag, value in info.items():
                        logger_val.log_scalar(tag, value, iterations)


def test(model, loader_test, device=torch.device('cuda'),
         dtype_x=None, dtype_y=None):
    """Test on singlet without any modification on data.

    Args: 
        model (:obj:`torch.nn.Module`): A PyTorch Module for a model to be
            trained.
        loader_test (:obj:`torch.utils.data.DataLoader`): DataLoader having
            test data.
        device (:obj:`torch.device`, optional): Device where training is being
            held. Default is CUDA.
        dtype_x (:obj:`dtype`): Data type of input data. Default is None.
        dtype_y (:obj:`dtype`): Data type of classifier. Default is None.

    Returns:
        Accuracy from the test result.
    """
    correct = 0
    total = 0
    acc = None
    with torch.no_grad():
        for x, y in loader_test:
            # For single image use below:
            if dtype_x is not None:
                x = x.to(device=device, dtype=dtype_x)
            else:
                x = x.to(device=device)
            if dtype_y is not None:
                y = y.to(device=device, dtype=dtype_y)
            else:
                y = y.to(device=device)
            out = model(x)
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        acc = float(correct) / total
        print('[{}/{}] Correct ({:.2f} %)'.format(correct, total, acc * 100))

    return acc
