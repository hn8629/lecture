import numpy as np

def dataset(seed,d=10,train_size=100,test_size=20):
    np.random.seed(seed)
    W_correct = np.random.randn(d)

    train = np.random.randn(train_size,d)
    train_label = np.dot(train,W_correct.reshape(-1,1))
    train_label[train_label >= 0] = 1
    train_label[train_label < 0] = -1
    train = (train,train_label)
    
    test = np.random.randn(test_size,d)
    test_label = np.dot(test,W_correct.reshape(-1,1))
    test_label[test_label >= 0] = 1
    test_label[test_label < 0] = -1
    test = (test,test_label)

    return (train,test)
