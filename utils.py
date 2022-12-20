def evaluate_accuracy(y, y_hat):
    indices = y_hat.argmax(axis=1)
    return (y == indices.type(y.dtype)).sum()/len(y)