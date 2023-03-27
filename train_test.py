import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def train(model, optimizer, train_loaders, device, batch_size, epoch, writer):
    torch.cuda.empty_cache()
    model.train()

    correct, total = 0.,0.
    total_acc = 0
    losses = []

    for train_ldr in train_loaders:
        for batch_idx, batch in enumerate(tqdm(train_ldr)):
            image = batch['image'].to(device)
            label = batch['label'].to(device)
            image_name = batch['image_name']
            is_ref = batch['is_ref']

            optimizer.zero_grad()

            output: torch.Tensor = model(image)

            loss = nn.CrossEntropyLoss()(output,label)
            loss.backward()

            optimizer.step()

            pred = output.argmax(dim=1,keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

            total += len(label)

            losses.append(loss.item())

    train_loss = float(np.mean(losses))
    total_acc = correct / ((batch_idx+1) * batch_size)

    print(f'Epoch {epoch}: Loss {train_loss:.6f}, Accuracy {100*total_acc:.2f}%')
    writer.add_scalar("Loss/train_batch", train_loss, epoch)
    writer.add_scalar("Accuracy/train_batch", total_acc, epoch)


def test(model, test_loader, device, batch_size, epoch, writer):
    torch.cuda.empty_cache()
    model.eval()

    correct, total = 0.,0.
    total_acc = 0
    losses = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            image = batch['image'].to(device)
            label = batch['label'].to(device)
            image_name = batch['image_name']
            is_ref = batch['is_ref']

            output: torch.Tensor = model(image)

            loss = nn.CrossEntropyLoss()(output,label)

            pred = output.argmax(dim=1,keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

            total += len(label)

            losses.append(loss.item())

    train_loss = float(np.mean(losses))
    total_acc = correct / ((batch_idx+1) * batch_size)

    print(f'Epoch {epoch}: Loss {train_loss:.6f}, Accuracy {100*total_acc:.2f}%')
    writer.add_scalar("Loss/train_batch", train_loss, epoch)
    writer.add_scalar("Accuracy/train_batch", total_acc, epoch)
