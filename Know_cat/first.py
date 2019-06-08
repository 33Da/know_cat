import numpy as np
#画图的库
import matplotlib.pyplot as plt
#h5py用来加载训练集
import h5py
#缩放图片
import skimage.transform as tf



#加载资源
def load_dataset():
    train_data=h5py.File("train_catvnoncat.h5","r")


    #提取特征,取出key值为train_set_x的所有值
    train_set_x_orig=np.array(train_data["train_set_x"][:])
    #提取标签
    train_set_y_orig = np.array(train_data["train_set_y"][:])

    test_data=h5py.File("test_catvnoncat.h5","r")
    #提取特征
    test_set_x_orig = np.array(test_data["test_set_x"][:])
    # 提取标签
    test_set_y_orig = np.array(test_data["test_set_y"][:])

    #1.有猫 2.无猫
    classes=np.array(test_data["list_classes"][:])
    #把维度（209，）改成（1，209）  shape[0]:一维大小，即行数   shape[1]:二维大小，即列数
    #reshape:改变矩阵结构 例reshape（2，4），把矩阵变成2行4列
    train_set_y_orig=train_set_y_orig.reshape((1,train_set_y_orig.shape[0]))
    #把维度（50，）改成（1，50）
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig ,train_set_y_orig ,test_set_x_orig,test_set_y_orig,classes


# 学习工具函数
def sigmoid(z):
    # 参数：
    # z-一个数值或者一个numpy数组
    # 返回值：
    # s-经过sigmoid计算后的值，在[0,1]范围内
    s=1/(1+np.exp(-z))
    return s

def initialize_with_zeros(dim):
    # 用于初始化权重数组w和偏置/阈值b
    # 参数：
    # dim- w的大小，一个特征对应一个权重
    # 返回值：
    # w- 权重数组
    # b-偏置bias
    w=np.zeros((dim,1))
    b=0
    return w,b

def propagate(w,b,x,y):
    #用于执行前向传播和反向传播
    # 参数：
    # w -- 权重数组
    # b --  偏置bias
    # x -- 图片的特征数据，维度（12288，1）
    # y -- 图片的对应标签，0或1，0是无猫，1是有猫，维度（1，209）
    # 返回值：
    # cost --成本
    # dw  --w的梯度
    # db  --b的梯度
    m=x.shape[1]

    #前向传播
    A=sigmoid(np.dot(w.T,x)+b)
    cost=-np.sum(y*np.log(A)+(1-y)*np.log(1-A))/m

    #反向传播
    dz=A-y
    dw=np.dot(x,dz.T)/m
    db=np.sum(dz)/m

    #将dw和db存到一个字典里面
    grads={
        "dw": dw,
        "db": db
    }
    return grads, cost

def optimize(w,b,x,y,num_iterations,learning_rate,print_cost=False):
    # 梯度下降更新参数w和b，达到越来越优化的目的

    # 参数：
    # w --权重数组，维度（12288,1)
    # b --偏置bias
    # x --图片的特征数据，维度（12288，209）
    # y --图片的对应标签，0无猫，1有猫
    # num_iterations --优化次数
    # learning_rate --学习步进，控制优化步进
    # print_cost --为true时，优化100次把成本cost打印出来
    #
    # 返回值：
    # params:优化后的w和b
    # costs：每优化100次，将成本记录下来，成本越小，参数越优化

    costs=[]

    for i in range(num_iterations):
        grads,cost= propagate(w,b,x,y)

        # 从字典中取出梯度
        dw=grads["dw"]
        db=grads["db"]

        #进行梯度下降，更新参数
        w=w-learning_rate*dw
        b=b-learning_rate*db

        #将成本记录下来
        if i%100 ==0:
            costs.append(cost)
            if print_cost:
                print("优化%i次后的成本为：%f"%(i,cost))

    params={
        "w":w,
        "b":b
    }
    return params,costs


def predict(w,b,x):
   # 参数：
   # w --权重数组，维度是（12288，1）
   # b --偏置bias
   # x --图片的特征数据，维度（12288，图片张数）

   # 返回值：
   # y_prediction --对每张图片的预测结果
   m=x.shape[1]
   y_predicition=np.zeros((1,m))

   #对图片进行预测
   a=sigmoid(np.dot(w.T,x)+b)

   # 上面得出的预测结果是小数形式，为了方便后面的显示，我们将其转换成0和1的形式（大于等于0.5就是1/有猫，小于0.5就是0/无猫）
   for i in range(a.shape[1]):
       if a[0,i]>=0.5:
           y_predicition[0,i]=1

   return y_predicition






#1.把数据出来
train_set_x_orig ,train_set_y ,test_set_x_orig,test_set_y,classes=load_dataset()
# 检验一下，看数据是否抽取成功
index=6
plt.imshow(train_set_x_orig[index])

print("标签为"+str(train_set_y[:,index])+",这是一个"+classes[np.squeeze(train_set_y[:,index])].decode("utf-8")+"图片")
plt.show()

# 查看各个数据的维度
print("train_set_x_orig.shape:"+str(train_set_x_orig.shape))
print("train_set_y.shape:"+str(train_set_y.shape))
print("test_set_x_orig.shape:"+str(test_set_x_orig.shape))
print("test_set_y.shape:"+str(test_set_y.shape))
# train_set_x_orig的维度含义是(样本数，图片宽，图片长，RGB三通道)



# 2.对读取的数据进行处理，把样本数和长宽提取出来
#训练集样本数
m_train=train_set_x_orig.shape[0]
# 测试集样本数
m_test=test_set_x_orig.shape[0]
# 每张图片的长/宽
num_px=test_set_x_orig.shape[1]

print("训练集样本数："+str(m_train))
print("测试集样本数："+str(m_test))
print("每张图片的长/宽："+str(num_px))

# 为了方便后面的矩阵运算，将样本数据扁平化和转置
train_set_x_flatten=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
#处理后的样本表示(图片数据，样本数量)

print("train_set_x_flatten shape:"+str(train_set_x_flatten.shape))
print("test_set_x_flatten shape:"+str(test_set_x_flatten.shape))

# 对数据进行标准化预处理，除以255，使所有值落在[0,1]之间，方便之后计算
train_set_x=train_set_x_flatten/255
test_set_x_=test_set_x_flatten/255




#3.构建神经网络
def model(x_train,y_train,x_test,y_test,num_iterations=2000,learning_rate=0.5,print_cost=False):
    # 参数：
    # x_train --训练图片，维度是（12288，209）
    # y_train--训练图片对应标签（1，209）
    # x_test--测试图片（12288，50）
    # y_test --测试图片对应的标签（1，50）
    # num_iterations --需要训练/优化多少次
    # learning_rate --学习步进，是我们用来控制优化步进的参数
    # print_cost --为true，每次优化100次就把成本cost打印出来
    #
    # 返回：
    # d --信息

    #初始化训练数据

    w,b=initialize_with_zeros(x_train.shape[0])


    #使用训练数据来训练/优化参数
    parameters, costs=optimize(w,b,x_train,y_train,num_iterations,learning_rate,print_cost)

    #从字典中分别取出训练好的w和b
    w=parameters["w"]
    b=parameters["b"]

    # 使用训练好的w和b对图片和测试图片进行预测
    y_prediction_train=predict(w,b,x_train)
    y_prediction_test=predict(w,b,x_test)

    # 打印出来预测准确率
    print("训练图片预测的准确率为：{}%".format(100-np.mean(np.abs(y_prediction_train-y_train))*100))
    print("测试图片预测的准确率为：{}%".format(100-np.mean(np.abs(y_prediction_test-y_test))*100))

    d={
        "costs":costs,
        "y_prediction_train":y_prediction_train,
        "y_prediction_test": y_prediction_test,
        "w":w,
        "b":b,
        "learning_rate":learning_rate,
        "num_iterations":num_iterations

    }
    return d

d=model(train_set_x,train_set_y,test_set_x_,test_set_y,num_iterations=2000,learning_rate=0.005,print_cost=True)


# 改变index，看哪些图片预测对了
# index=8
# plt.imshow(test_set_x_[:,index].reshape((num_px,num_px,3)))
# print("这张图的标签是"+str(test_set_y[0,index])+",预测结果是"+str(int(d["y_prediction_test"][0,index])))
# plt.show()

# 成本随训练次数变化情况
# costs=np.squeeze(d["costs"])
# plt.plot(costs)
# plt.ylabel('cost') #成本
# plt.xlabel("iterations") #训练次数
# plt.title("Learning rate ="+str(d["learning_rate"]))
# plt.show()

# 选择学习率learning_rate
# learning_rates=[0.01,0.001,0.0001]
# models={}
# for i in learning_rates:
#     print("学习率为："+str(i)+"时")
#     models[str(i)]=model(train_set_x,train_set_y,test_set_x_,test_set_y,num_iterations=1500,learning_rate=i,print_cost=False)
#     print("\n"+"------------------------------"+"\n")
# for i in learning_rates:
#     plt.plot(np.squeeze(models[str(i)]["costs"]),label=str(models[str(i)]["learning_rate"]))
#
# plt.ylabel("cost")
# plt.xlabel("iterations")
# legend=plt.legend(loc='upper center',shadow=True)
# frame=legend.get_frame()
# frame.set_facecolor("0.9")
# plt.show()

#预测自己的图片
my_image="5.png"

fname="image/"+my_image
image=np.array(plt.imread(fname))
my_image=tf.resize(image,(num_px,num_px),mode="reflect").reshape((1,num_px*num_px*3)).T
my_predicted_image=predict(d["w"],d["b"],image)
plt.imshow(image)
print("预测结果为："+str(int(np.squeeze(my_predicted_image))))
plt.show()

