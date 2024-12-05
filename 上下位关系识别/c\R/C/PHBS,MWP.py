from gensim.models import KeyedVectors
import torch
import math
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
import warnings
import numpy as np
from hyperlib.embedding.treerep import treerep
from hyperlib.embedding.sarkar import sarkar_embedding
from sklearn.cluster import KMeans as BaseKMeans
from decimal import Decimal
import torch
import torch.nn as nn
import geoopt
from sklearn.metrics import accuracy_score, precision_score, recall_score

def poincare_subtraction(x, y, c):
    x2 = np.sum(x * x, axis=-1, keepdims=True)
    y2 = np.sum(y * y, axis=-1, keepdims=True)
    xy = np.sum(x * y, axis=-1, keepdims=True)
    num = (1 - 2 * c * xy + c * y2) * x + (1 + c * x2) * y
    denom = 1 - 2 * c * xy + c ** 2 * x2 * y2
    return num / np.maximum(denom, np.finfo(float).eps)


def poincare_add(x, y, c):
    x2 = np.sum(x * x, axis=-1, keepdims=True)
    y2 = np.sum(y * y, axis=-1, keepdims=True)
    xy = np.sum(x * y, axis=-1, keepdims=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / np.maximum(denom, np.finfo(float).eps)


def poincare_transform(M, x):

    Mx = np.dot(M, x)
    norm_Mx = np.linalg.norm(Mx)
    norm_x = np.linalg.norm(x)

    # 计算 tanh 和 actanh
    actanh_x_norm = np.arctanh(norm_x)
    transform_value = (norm_Mx / norm_x) * actanh_x_norm

    # 计算变换结果
    transformed_x = np.tanh(transform_value) * (Mx / norm_Mx)

    return transformed_x



def qianru(x, c):
    tree = treerep(x, return_networkx=True)

    # embed the tree in 2D hyperbolic space
    root = 0
    embedding = sarkar_embedding(tree, root, tau=0.5)
    return embedding

def poincare_distance(x, y, c):
    sqrt_c = np.sqrt(c)
    x2 = np.sum(x * x, axis=-1, keepdims=True)
    y2 = np.sum(y * y, axis=-1, keepdims=True)
    xy = np.sum(x * y, axis=-1, keepdims=True)
    c1 = 1 - 2 * c * xy + c * y2
    c2 = 1 - c * x2
    num = np.sqrt((c1 ** 2) * x2 + (c2 ** 2) * y2 - (2 * c1 * c2) * xy)
    denom = 1 - 2 * c * xy + c ** 2 * x2 * y2

    # 使用 np.maximum 确保 denom 不为零，防止除以零的情况
    pairwise_norm = num / np.maximum(denom, np.finfo(float).eps)

    # 使用 np.clip 确保输入 arctanh 的值在 (-1, 1) 范围内
    pairwise_norm = np.clip(pairwise_norm, -1 + np.finfo(float).eps, 1 - np.finfo(float).eps)

    dist = np.arctanh(sqrt_c * pairwise_norm)
    return 2 * dist / sqrt_c


def update_centroids(data, labels, k):
    """
    更新聚类中心，对NaN进行处理。
    """
    centroids = []
    for i in range(k):
        cluster_points = data[labels == i]
        # 使用平均向量作为新中心的简化处理
        if len(cluster_points) == 0:  # 避免空聚类
            continue
        centroid = np.mean(cluster_points, axis=0)
        if np.isnan(centroid).any():  # 检查NaN并处理
            centroid = np.nan_to_num(centroid, nan=0.0)  # 将NaN替换为0，或选择其他合理的默认值
        centroids.append(centroid / np.linalg.norm(centroid))  # 保持在Poincaré球内
    return np.array(centroids)


class CustomKMeans(BaseKMeans):
    def fit(self, X, y=None):
        # 使用父类的 fit 方法来训练模型
        super().fit(X, y)
        return self

    def distances_to_centers(self, X):
        # 计算数据点到聚类中心的距离
        distances = poincare_distance(X, self.cluster_centers_,1)
        return distances




def load_pairs(path):
    pairs=set()
    file = open(path,encoding="utf-8")
    while 1:
        line = file.readline()
        if not line:
            break
        line = line.replace('\n', '')
        str = line.split('\t')
        hypo = str[0]
        hyper = str[1]
        pairs.add((hypo,hyper))
    file.close()
    return pairs

def cluster_embeddings(pairs, en_model):
    global n_clusters, n_embeddings
    new_pairs=list()
    new_pairs2=list()
    temp=np.zeros(shape=(len(pairs),n_embeddings))
    temp2=np.zeros(shape=(len(pairs),n_embeddings))
    i=0
    for hypo,hyper in pairs:
        temp[i]=poincare_subtraction(qianru(en_model[hyper],1),qianru(en_model[hypo],1),1)
        temp2[i] = poincare_subtraction(qianru(en_model[hypo], 1),qianru(en_model[hyper], 1), 1)
        i=i+1
    estimator = CustomKMeans(n_clusters)
    estimator.fit(temp)
    centroids = estimator.cluster_centers_
    estimator2 = CustomKMeans(n_clusters)
    estimator2.fit(temp2)
    centroids2 = estimator2.cluster_centers_

    for hypo,hyper in pairs:
        vector=poincare_subtraction(qianru(en_model[hyper],1),qianru(en_model[hypo],1))
        vector2=poincare_subtraction(qianru(en_model[hyper],1),qianru(en_model[hypo],1))
        weights = np.zeros(shape=n_clusters)
        weights2 = np.zeros(shape=n_clusters)
        for i in range(n_clusters):
            weights[i]=np.power(math.e, poincare_distance(vector.reshape(1, -1),centroids[i].reshape(1, -1),1))
            weights2[i]=np.power(math.e, poincare_distance(vector2.reshape(1, -1),centroids2[i].reshape(1, -1),1))
        new_pairs.append((hypo,hyper,weights))
        new_pairs2.append((hypo,hyper,weights2))
    return centroids,centroids2, new_pairs, new_pairs2

def learn_dual_projection(en_model, new_pairs,new_pairs2, cluster_index):

    global n_embeddings
    global sparsity
    # w: hyper embeddings, v: hypo embeddings
    B1 = np.zeros(shape=(n_embeddings, n_embeddings))
    B2 = np.zeros(shape=(n_embeddings, n_embeddings))
    M1=np.zeros(shape=(n_embeddings,n_embeddings))
    M2=np.zeros(shape=(n_embeddings,n_embeddings))
    for hypo, hyper, weights in new_pairs:
        a = qianru(en_model[hyper],1).reshape(n_embeddings, 1)
        b = qianru(en_model[hypo],1).reshape(1, n_embeddings)
        M1= poincare_add(M1 , np.matmul(a, b),1)

    for hypo, hyper, weights in new_pairs2:
        aa = qianru(en_model[hyper],1).reshape(1,n_embeddings)
        bb = qianru(en_model[hypo],1).reshape(n_embeddings,1)
        M2 = poincare_add(M2 , np.matmul(bb, aa),1)

    for hypo, hyper, weights in new_pairs:
        #np.random.seed(50)  # 设置随机数状态
        temp_weights = np.zeros((n_embeddings, n_embeddings))
        num_ones = int(n_embeddings * n_embeddings * sparsity)
        indices = np.random.choice(n_embeddings * n_embeddings, num_ones, replace=False)
        temp_weights[np.unravel_index(indices, (n_embeddings, n_embeddings))] =   weights[cluster_index]
        a = qianru(en_model[hyper],1).reshape(n_embeddings, 1)
        b = qianru(en_model[hypo],1).reshape(1, n_embeddings)
        B1 = poincare_add(B1 , np.multiply(temp_weights, np.matmul(M2,np.matmul(a, b))),1)
        M_1 = np.eye(n_embeddings, n_embeddings)
        B1 = poincare_add(B1 , M_1,1)

    for hypo, hyper, weights in new_pairs2:
        temp_weights = np.zeros((n_embeddings, n_embeddings))
        num_ones = int(n_embeddings * n_embeddings * sparsity)
        indices = np.random.choice(n_embeddings * n_embeddings, num_ones, replace=False)
        temp_weights[np.unravel_index(indices, (n_embeddings, n_embeddings))] =   weights[cluster_index]
        aa = qianru(en_model[hyper],1).reshape(1,n_embeddings)
        bb = qianru(en_model[hypo],1).reshape(n_embeddings,1)
        B2 = poincare_add(B2 , np.multiply(temp_weights, np.matmul(M1,np.matmul(bb, aa))),1)
        M_1 = np.eye(n_embeddings, n_embeddings)
        B2= poincare_add(B2 ,M_1,1)

    return B1, B2  # Return both rotation matrices

def learn_projections(en_model, new_pairs, new_pairs2):
    global n_clusters
    projection_matrices1=list()
    projection_matrices2=list()
    for i in range(n_clusters):
        R1, R2 = learn_dual_projection(en_model, new_pairs, new_pairs2, cluster_index=i)
        projection_matrices1.append(R1)
        projection_matrices2.append(R2)
    return projection_matrices1, projection_matrices2

def generate_features(en_model, pairs, positive_projections1,positive_projections2, positive_centroids,positive_centroids2, negative_projections1,negative_projections2, negative_centroids, negative_centroids2):
    global n_clusters
    features=list()
    for hypo, hyper in pairs:
        vector = poincare_subtraction(qianru(en_model[hyper],1),qianru(en_model[hypo],1),1)
        vector2 = poincare_subtraction(qianru(en_model[hypo],1),qianru(en_model[hyper],1),1)
        positive_weights = np.zeros(shape=n_clusters)
        positive_weights2 = np.zeros(shape=n_clusters)
        negative_weights = np.zeros(shape=n_clusters)
        negative_weights2 = np.zeros(shape=n_clusters)
        for i in range(n_clusters):
            positive_weights[i] = np.power(math.e, poincare_distance(vector.reshape(1, -1), positive_centroids[i].reshape(1, -1),1))
            negative_weights[i] = np.power(math.e, poincare_distance(vector.reshape(1, -1), negative_centroids[i].reshape(1, -1),1))
        pos_concat=np.zeros(shape=(n_clusters,n_embeddings))
        neg_concat=np.zeros(shape=(n_clusters,n_embeddings))
        for i in range(n_clusters):
            pos_f_i=poincare_transform(positive_weights[i],poincare_subtraction(poincare_transform((positive_projections1[i], qianru(en_model[hypo],1)),poincare_transform((positive_projections2[i], qianru(en_model[hyper],1)),1))))
            neg_f_i=poincare_transform(negative_weights[i],poincare_subtraction(poincare_transform((negative_projections1[i], qianru(en_model[hypo],1)),poincare_transform((negative_projections2[i], qianru(en_model[hyper],1)),1))))
            pos_concat[i]=pos_f_i
            neg_concat[i]=neg_f_i
        pos_concat=pos_concat.reshape(n_embeddings*n_clusters, order='C')
        neg_concat=neg_concat.reshape(n_embeddings*n_clusters, order='C')
        all_features= np.concatenate((pos_concat, neg_concat), axis=None)
        features.append(all_features)
    return features

class PoincareClassifier(nn.Module):
    def __init__(self, input_dim,  output_dim):
        super(PoincareClassifier, self).__init__()
        self.hidden = nn.Linear(input_dim, 128)  # 隐藏层
        self.relu = nn.ReLU()                            # ReLU 激活函数
        self.linear = nn.Linear(128, output_dim)  # 输出层
        self.manifold = geoopt.manifolds.PoincareBall(c=1.0)

    def forward(self, x):
        x = self.hidden(x)                            # 线性变换
        x = self.relu(x)                              # 激活函数
        x = self.linear(x)                            # 输出层
        x = self.manifold.projx(x)                    # 映射到 Poincaré 球上
        return x


def train_classifier(pos_features, neg_features):
    # for simplicity, we use a part of the data for training and the rest for testing
    # please refer to the paper for detailed evaluation methods
    dim=2*n_clusters*n_embeddings
    pos_len = len(pos_features)
    neg_len = len(neg_features)
    pos_train=list()
    neg_train = list()
    for i in range(0,pos_len):
        pos_train.append(pos_features[i])
    for i in range(0, neg_len):
        neg_train.append(neg_features[i])

    train_data = np.zeros(shape=(len(pos_train)+len(neg_train), dim))
    train_labels = []
    for i in range(0,len(pos_train)):
        train_data[i]=pos_train[i]
        train_labels.append(1)
    for i in range(0,len(neg_train)):
        train_data[i+len(pos_train)]=neg_train[i]
        train_labels.append(0)

    train_data_tensor = torch.from_numpy(train_data).float()  # 转换为浮点型
    my_array = np.array(train_labels)
    train_labels_tensor = torch.from_numpy(my_array).long()  # 转换为长整型

    net = PoincareClassifier(dim, 2)

    # 定义损失函数（交叉熵损失）
    criterion = nn.CrossEntropyLoss()

    # 初始化优化器
    optimizer = geoopt.optim.RiemannianAdam(net.parameters(), lr=0.01)

    # 假设进行 100 个训练轮次
    for epoch in range(100):
        optimizer.zero_grad()  # 清空梯度
        output = net(train_data_tensor)  # 前向传播
        loss = criterion(output, train_labels_tensor)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        if epoch % 10 == 0:  # 每 10 轮打印一次损失
            print(f'Epoch [{epoch}/100], Loss: {loss.item():.4f}')
    return net

def test_classifier(pos_features, neg_features, cls):
    global n_clusters, n_embeddings
    dim = 2 * n_clusters * n_embeddings
    test_data = np.zeros(shape=(len(pos_features) + len(neg_features), dim))
    test_labels = []
    for i in range(0, len(pos_features)):
        test_data[i] = pos_features[i]
        test_labels.append(1)
    for i in range(0, len(neg_features)):
        test_data[i + len(pos_features)] = neg_features[i]
        test_labels.append(0)
    my_array= np.array(test_labels)
    test_data_tensor = torch.from_numpy(test_data).float()  # 转换为浮点型
    true_labels= torch.from_numpy(my_array).long()  # 转换为长整型

    with torch.no_grad():  # 在测试时不计算梯度
        test_data = torch.randn(len(test_labels), dim)  # 5 个测试样本
        test_output = cls(test_data_tensor)  # 前向传播
        test_predictions = torch.argmax(test_output, dim=1)  # 获取预测类别
        accuracy = accuracy_score(true_labels.numpy(), test_predictions.numpy())
        precision = precision_score(true_labels.numpy(), test_predictions.numpy(), average='weighted')
        recall = recall_score(true_labels.numpy(), test_predictions.numpy(), average='weighted')

        print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')




emb=[]
clu=[]
spa=[]
for i in emb:
    for j in clu:
        for k in spa:
            global n_clusters
            n_clusters = j
            global n_embeddings
            n_embeddings = i
            global sparsity
            sparsity = k
            warnings.filterwarnings("ignore")
            print('load fast text model...')
            en_model = KeyedVectors.load_word2vec_format('')
            # print('model load successfully')
            positive_pairs_train = load_pairs('')
            negative_pairs_train = load_pairs('')
            print('data load successfully')
            positive_pairs_test = load_pairs('')
            negative_pairs_test = load_pairs('')

            datad = list(positive_pairs_train) + list(negative_pairs_train) + list(positive_pairs_test) + list(
                negative_pairs_test)
            # 训练 Poincaré Embeddings 模型
            # projection learning
            pos_centroids, pos_centroids2, pos_new_pairs, pos_new_pairs2 = cluster_embeddings(positive_pairs_train,
                                                                                              en_model)
            print('cluster pos embeddings successfully')
            neg_centroids, neg_centroids2, neg_new_pairs, neg_new_pairs2 = cluster_embeddings(negative_pairs_train,
                                                                                              en_model)
            print('cluster neg embeddings successfully')
            pos_projections1, pos_projections2 = learn_projections(en_model, pos_new_pairs, pos_new_pairs2)
            print('learn pos projections successfully')
            neg_projections1, neg_projections2 = learn_projections(en_model, neg_new_pairs, neg_new_pairs2)
            print('learn neg projections successfully')

            # feature generation
            pos_features = generate_features(en_model, positive_pairs_train, pos_projections1, pos_projections2,
                                             pos_centroids,
                                             pos_centroids2,
                                             neg_projections1, neg_projections2, neg_centroids, neg_centroids2)
            print('positive features generation successfully')
            neg_features = generate_features(en_model, negative_pairs_train, pos_projections1, pos_projections2,
                                             pos_centroids,
                                             pos_centroids2,
                                             neg_projections1, neg_projections2, neg_centroids, neg_centroids2)
            print('negative features generation successfully')

            # classifier training
            cls = train_classifier(pos_features, neg_features)

            print('data load successfully')
            pos_features = generate_features(en_model, positive_pairs_test, pos_projections1, pos_projections2,
                                             pos_centroids,
                                             pos_centroids2,
                                             neg_projections1, neg_projections2, neg_centroids, neg_centroids2)
            print('positive features generation successfully')
            neg_features = generate_features(en_model, negative_pairs_test, pos_projections1, pos_projections2,
                                             pos_centroids,
                                             pos_centroids2,
                                             neg_projections1, neg_projections2, neg_centroids, neg_centroids2)
            print('negative features generation successfully')
            test_classifier(pos_features, neg_features, cls)
