
import numpy as np
def evaluate(model, criterion, metric, data_loader, phase="dev"):

    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch['input_ids'], batch['token_type_ids'], batch['labels']
        logits = model(input_ids=input_ids, token_type_ids=token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval {} loss: {:.5}, accu: {:.5}".format(phase,
                                                    np.mean(losses), accu))
    model.train()
    metric.reset()
    return accu, np.mean(losses)