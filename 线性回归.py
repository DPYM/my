import numpy as np

def LinearRegression(lr=0.01,iterate=1000):
    print('---加载数据---')
    data=load_data_csv('data.csv',split=',',data_type=np.float32)
    print('---加载成功---')

    x=data[:,0:-2]
    y=data[:,-1]
    m=len(y)
    col=data.shape[1]

    print('---标准化处理---')
    x,mu,sigma=Normalize(x=x)
    w=np.zeros(x.shape[1])
    b=0
    w,b=grad(x,y,w,b,lr,iterate,m)
    
    return mu,sigma


#加载文件
def load_data_csv(file_name,split,data_type):
    return np.loadtxt(file_name,delimiter=split,dtype=data_type)

#定义归一化
def Normalize(x):
    mu=np.mean(x,0)
    sigma=np.std(x,0)

    sigma[sigma==0]=1
    x_norm=(x-mu)/sigma
    return x_norm,mu,sigma

#梯度下降
def grad(x,y,w,b,lr,iterate,m):
    for epoch in range(iterate):
        y_pred=x.dot(w)+b
        loss=(1/m)*sum((y-y_pred)**2)
        dw=(2/m)*x.T.dot(y_pred-y)
        db=(2/m)*np.sum(y_pred-y)
        w=w-lr*dw
        b=b-lr*db

        if (epoch+1)%100==0:
            print(f'第{epoch+1}次损失为{loss:.4f}')
    return w,b

def test():
    mu,sigma=LinearRegression(0.01,1000)
    print(f'均值{mu}，标准差{sigma}')

if __name__=='__main__':
    test()