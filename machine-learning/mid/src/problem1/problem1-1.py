import numpy as np
import dataset
import matplotlib.pyplot as plt

d = 10
batch_size = 10
lr = 0.01
lam = 0.0002
wd = lam/2
epoch = 50
train_size = 100
test_size = 20
iteration = epoch*train_size/batch_size

def predict(w,data):
    data_size = data[0].shape[0]
    pred = np.dot(data[0],w.reshape(-1,1))
    pred[pred >= 0] = 1
    pred[pred < 0] = -1
    counter = np.zeros_like(pred)
    counter[pred == data[1]] = 1
    acc = np.sum(counter)/counter.shape[1]
    return acc/data_size

def calc_J(x,y,w):
    exp = np.exp(-y*np.dot(x,w.reshape(-1,1)))
    return np.sum(np.log(1+exp),axis=0) + lam*np.dot(w,w)

def grad(x,y,w):
    exp = np.exp(-y*np.dot(x,w.reshape(-1,1)))
    return  np.sum(-y*exp/(1+exp)*x,axis=0)+wd*w




train,test = dataset.dataset(10,d,train_size,test_size)

train_batches = (np.split(train[0],train_size/batch_size),np.split(train[1],train_size/batch_size))

np.random.seed(20)
W = np.random.randn(d)

print('[before training]  train_acc: %f'%predict(W,train))
print('[before training]  test_acc: %f'%predict(W,test))


j_list = []
for i in range(epoch):
    for x,y in zip(train_batches[0],train_batches[1]):
        W -= lr*grad(x,y,W)
        j_list.append(calc_J(train[0],train[1],W))
    print('epoch: %d  J: %f'%(i+1,calc_J(train[0],train[1],W)))

print('[after training]  train_acc: %f'%predict(W,train))
print('[after training]  test_acc: %f'%predict(W,test))

plt.plot(j_list,'-')

plt.grid()

plt.title("J(w)")
plt.xlabel("iteration")
plt.ylabel("J(w)")
xt = [(i+1)*100-1 for i in range(int(iteration/100))]
xt.insert(0,0)
xt = np.array(xt)
plt.xticks(xt,xt+1)
#plt.show()
plt.savefig('j-sgd.png')


