import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def train(epoch, data_loader, model, optimizer, device, hyper, recorder, scaler=None):
    recorder.info('------starting {} epoch training------'.format(epoch))
    model.train()
    train_loss = 0.0
    total_targets = []
    total_pred_targets = []

    for i, batch in enumerate(data_loader, 1):
        optimizer.zero_grad()
        batch.to(device=device)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss, targets, pred_targets = model(batch, device, hyper)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, targets, pred_targets = model(batch, device, hyper)
            loss.backward()
            optimizer.step()
        total_targets.extend(targets.tolist())
        total_pred_targets.extend(pred_targets.tolist())
        train_loss += loss.cpu().item()
    accuracy = accuracy_score(total_targets, total_pred_targets)
    f1 = f1_score(total_targets, total_pred_targets)
    recall = recall_score(total_targets, total_pred_targets)
    precision = precision_score(total_targets, total_pred_targets)
    log = "train epoch[{}] end! loss: {:.4f} acc: {:.2f}% f1: {:.2f}% recall: {:.2f}% prec: {:.2f}%".format(
        epoch, train_loss / i, accuracy * 100, f1 * 100, recall * 100, precision * 100)
    recorder.info(log)
    return train_loss / i, accuracy, f1, recall, precision

def valid(epoch, data_loader, model, device, hyper, recorder):
    recorder.info('------starting {} epoch valid------'.format(epoch))
    model.eval()
    valid_loss = 0.0
    total_targets = []
    total_pred_targets = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader, 1):
            batch.to(device=device)
            loss, targets, pred_targets = model(batch, device, hyper)
            total_targets.extend(targets.tolist())
            total_pred_targets.extend(pred_targets.tolist())
            valid_loss += loss.cpu().item()
    accuracy = accuracy_score(total_targets, total_pred_targets)
    f1 = f1_score(total_targets, total_pred_targets)
    recall = recall_score(total_targets, total_pred_targets)
    precision = precision_score(total_targets, total_pred_targets)
    log = "valid epoch[{}] end! loss: {:.4f} acc: {:.2f}% f1: {:.2f}% recall: {:.2f}% prec: {:.2f}%".format(
        epoch, valid_loss / i, accuracy * 100, f1 * 100, recall * 100, precision * 100)
    recorder.info(log)
    return valid_loss / i, accuracy, f1, recall, precision

def test(data_loader, model, device, hyper, recorder):
    recorder.info('------starting test------')
    model.eval()
    test_loss = 0.0
    total_targets = []
    total_pred_targets = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader, 1):
            batch.to(device=device)
            loss, targets, pred_targets = model(batch, device, hyper)
            total_targets.extend(targets.tolist())
            total_pred_targets.extend(pred_targets.tolist())
            test_loss += loss.cpu().item()
    accuracy = accuracy_score(total_targets, total_pred_targets)
    f1 = f1_score(total_targets, total_pred_targets)
    recall = recall_score(total_targets, total_pred_targets)
    precision = precision_score(total_targets, total_pred_targets)
    log = "test end! loss: {:.4f} acc: {:.2f}% f1: {:.2f}% recall: {:.2f}% prec: {:.2f}%".format(
        test_loss / i, accuracy * 100, f1 * 100, recall * 100, precision * 100)
    recorder.info(log)
    return test_loss / i, accuracy, f1, recall, precision
