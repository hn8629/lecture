import numpy as np
import dataset
import matplotlib.pyplot as plt

d = 10
lr = 0.01
lam = 0.01
iteration = 50
train_size = 100
test_size = 20

def predict(w,data):
    data_size = data[0].shape[0]
    pred = np.dot(data[0],w.reshape(-1,1))
    pred[pred >= 0] = 1
    pred[pred < 0] = -1
    counter = np.zeros_like(pred)
    counter[pred == data[1]] = 1
    acc = np.sum(counter)/counter.shape[1]
    return acc/data_size


def dual_lagrange(alpha,K):
    return -1/(4*lam)*np.dot(np.dot(alpha,K),alpha.reshape(-1,1)) + np.sum(alpha)

def calc_K(x,y):
    k = x*y
    return np.dot(k,k.transpose())

def calc_w(x,y,alpha):
    return np.sum(x*y*alpha.reshape(-1,1),axis=0)

def calc_J(x,y,w):
    return np.sum(np.max(1-y*np.dot(x,w.reshape(-1,1)),0))+lam*np.dot(w,w)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

train,test = dataset.dataset(10,d,train_size,test_size)

np.random.seed(20)
alpha = np.random.randn(train_size)

K = calc_K(train[0],train[1])

W = calc_w(train[0],train[1],alpha)
print('[before training] train acc: %f, test_acc: %f'%(predict(W,train),predict(W,test)))

dl_list = []
j_list = []

for i in range(iteration):
    alpha = softmax(alpha - lr*(1/(2*lam)*np.dot(alpha,K)-1))
    W = calc_w(train[0],train[1],alpha)
    j = calc_J(train[0],train[1],W)
    dl = dual_lagrange(alpha,K)
    j_list.append(j)
    dl_list.append(dl)
    print('iteration: %d  J: %f, Dual Lagrange: %f'%(i+1,j,dl))

print('[after training] train acc: %f, test_acc: %f'%(predict(W,train),predict(W,test)))


plt.plot(j_list,'-')

plt.grid()

plt.title("J(w)")
plt.xlabel("iteration")
plt.ylabel("J(w)")
xt = [(i+1)*10-1 for i in range(int(iteration/10))]
xt.insert(0,0)
xt = np.array(xt)
plt.xticks(xt,xt+1)
#plt.show()
plt.savefig('j.png')

plt.close()

plt.plot(dl_list,'-')

plt.grid()

plt.title("Dual Lagrange function")
plt.xlabel("iteration")
plt.ylabel("Dual Lagrange function")
xt = [(i+1)*10-1 for i in range(int(iteration/10))]
xt.insert(0,0)
xt = np.array(xt)
plt.xticks(xt,xt+1)
#plt.show()
plt.savefig('dlf.png')
