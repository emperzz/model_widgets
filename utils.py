def evaluate_accuracy(y, y_hat):
    values, indices = y_hat.max(dim=1)
    return (y == indices).sum()/len(y)