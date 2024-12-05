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

from gensim.models import KeyedVectors
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
    temp=np.zeros(shape=(len(pairs),n_embeddings))
    i=0
    for hypo,hyper in pairs:
        temp[i]=poincare_subtraction(qianru(en_model[hyper],1),qianru(en_model[hypo],1),1)
        i=i+1
    estimator = CustomKMeans(n_clusters)
    estimator.fit(temp)
    centroids = estimator.cluster_centers_

    for hypo,hyper in pairs:
        vector=poincare_subtraction(qianru(en_model[hyper],1),qianru(en_model[hypo],1))
        weights = np.zeros(shape=n_clusters)
        for i in range(n_clusters):
            weights[i]=np.power(math.e, poincare_distance(vector.reshape(1, -1),centroids[i].reshape(1, -1),1))
        new_pairs.append((hypo,hyper,weights))
    return centroids, new_pairs


def learn_single_projection(en_model, new_pairs, cluster_index):
    global n_embeddings
    #w: hyper embeddings, v: hypo embeddings
    B=np.zeros(shape=(n_embeddings,n_embeddings))
    for hypo,hyper,weights in new_pairs:
        temp_weights = np.zeros((n_embeddings, n_embeddings))
        num_ones = int(n_embeddings * n_embeddings * sparsity)
        indices = np.random.choice(n_embeddings * n_embeddings, num_ones, replace=False)
        temp_weights[np.unravel_index(indices, (n_embeddings, n_embeddings))] =   weights[cluster_index]
        a = qianru(en_model[hyper],1).reshape(n_embeddings, 1)
        b = qianru(en_model[hypo],1).reshape(1, n_embeddings)
        M_1 = np.eye(n_embeddings, n_embeddings)
        B = poincare_add(B , np.multiply(temp_weights, np.matmul(M_1,np.matmul(a, b))),1)
        B = poincare_add(B , M_1,1)

    return B


def learn_projections(en_model, new_pairs):
    global n_clusters
    projection_matrices=list()
    for i in range(n_clusters):
        R=learn_single_projection(en_model,new_pairs,cluster_index=i)
        projection_matrices.append(R)
    return projection_matrices

def xia(en_model, positive_projections1):
    # 获取所有词汇和对应的向量
    all_vectors = {word: en_model[word] for word in en_model.key_to_index.keys()}
    xia_vectors = {}
    # 打印所有的词汇及其向量
    for word, vector in all_vectors.items():
        pos_concat = np.zeros(shape=(n_clusters, n_embeddings))
        for i in range(n_clusters):
            pos_concat[i] = poincare_transform(positive_projections1[i], vector)
        xia_vectors[word] = pos_concat.reshape(n_embeddings * n_clusters, order='C')
    return xia_vectors

def shang(en_model, positive_projections2):
    # 获取所有词汇和对应的向量
    all_vectors = {word: en_model[word] for word in en_model.key_to_index.keys()}
    shang_vectors = {}
    # 打印所有的词汇及其向量
    for word, vector in all_vectors.items():
        pos_concat = np.zeros(shape=(n_clusters, n_embeddings))
        for i in range(n_clusters):
            pos_concat[i] = vector
        shang_vectors[word]= pos_concat.reshape(n_embeddings * n_clusters, order='C')
    return shang_vectors



class HierarchicalModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(HierarchicalModel, self).__init__()
        self.W = nn.Linear(input_dim, hidden_dim)  # 可训练参数

    def forward(self, s_vector, t_vector):

        # 线性变换
        s_transformed = self.W(s_vector)  # 映射到上位词空间

        # 计算相关性得分
        score = torch.matmul(s_transformed, t_vector.T)  # [batch_size, batch_size]
        return score


def info_nce_loss(scores, pos_indices, neg_indices):
    batch_size = scores.size(0)
    temperature = 0.07

    # 获取正样本的得分
    pos_scores = scores[torch.arange(batch_size), pos_indices]  # [batch_size]
    # 获取负样本的得分
    neg_scores = scores[torch.arange(batch_size).unsqueeze(1), neg_indices]  # [batch_size, num_neg_samples]
    # 计算损失
    loss = -torch.mean(torch.log(torch.exp(pos_scores/temperature) /  torch.sum(torch.exp(neg_scores/temperature), dim=1)))

    return loss

class HierarchicalDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)  # 读取CSV文件

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        hyponym = self.data.iloc[idx]['hyponym']  # 获取下位词
        hypernym = self.data.iloc[idx]['hypernym']  # 获取上位词

        return hyponym, hypernym

    # 假设数据存储在 data.csv 中，包含 'hyponym' 和 'hypernym' 列


def train(data,num_epochs):
    dataset = HierarchicalDataset(data)
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True)
    model = HierarchicalModel(n_embeddings*n_clusters,n_embeddings*n_clusters)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for hyponym_batch, hypernym_batch in dataloader:
            x=[]
            for i in hyponym_batch:
                x.append(xia_vec[i])
            x = torch.tensor(x, dtype=torch.float)
            s=[]
            for i in hypernym_batch:
                s.append(shang_vec[i])
            s = torch.tensor(s, dtype=torch.float)
            optimizer.zero_grad()  # 清空梯度

            # 前向传播
            scores = model(x, s)
            batch_size = x.size(0)
            pos_indices = torch.arange(batch_size)  # 正样本索引

            # 生成负样本索引，排除对应的上位词
            neg_indices = []
            for i in range(batch_size):
                # 生成当前样本的所有可能负例，排除自身和表中出现的负例
                cur_neg_indices = [j for j in range(batch_size)if (hyponym_batch[i], hypernym_batch[j]) not in all]

                # 如果负例数量不足，随机补全至 batch_size 个负例
                while len(cur_neg_indices) < batch_size:
                    rand_index = random.choice([j for j in range(batch_size) if j not in cur_neg_indices])
                    cur_neg_indices.append(rand_index)

                # 只取 batch_size 个负例
                cur_neg_indices = cur_neg_indices[:batch_size]

                neg_indices.extend(cur_neg_indices)  # 扩展到总的负例列表

            neg_indices = torch.tensor(neg_indices).reshape(batch_size, -1)  # 重塑为二维张量
            # 计算损失
            loss = info_nce_loss(scores, pos_indices, neg_indices)

            # 反向传播
            loss.backward()
            optimizer.step()  # 更新参数

            total_loss += loss.item()

        #avg_loss = total_loss / len(dataloader)
        #print(f"Epoch [{epoch + 1}/{num_epochs}]")
    return model
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
def xunlina(zheng, fu,top_k):
    train_data = np.zeros(shape=(top_k, n_embeddings * n_clusters))
    train_labels = np.zeros(shape=(top_k, 1))
    for i in range(0, len(zheng)):
        te = xia_vec[zheng[i][0]] - shang_vec[zheng[i][1]]
        train_data[i] = te
        train_labels[i] = zheng[i][2]
    for i in range(0, top_k-len(zheng)):
        te = xia_vec[fu[i][0]] - shang_vec[fu[i][1]]
        train_data[i] = te
        train_labels[i] = fu[i][2]

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

def jingpai(model, hyponym_data, hypernym_data, true_hypernyms, top_k):
    top_ka=[]
    top_kb=[]

    model.eval()
    with torch.no_grad():
        x = []
        for i in hyponym_data:
            x.append(xia_vec[i])
        x = torch.tensor(x, dtype=torch.float)
        s = []
        for i in hypernym_data:
            s.append(shang_vec[i])
        s = torch.tensor(s, dtype=torch.float)

        # 前向传播
        scores = model(x, s)
        top_k_scores, top_k_indices = torch.topk(scores, k=top_k, dim=1)

    for i in range(len(hyponym_data)):
        top_indices = top_k_indices[i].cpu().numpy()
        top_kb.append(top_indices)
        top_hypernyms2 = [hypernym_data[idx] for idx in top_indices]
        zheng=[]
        fu=[]
        dks = []
        dui22=[]
        for j in top_hypernyms2 :
            dss = (hyponym_data[i], j)
            dui22.append(dss)
            te = xia_vec[hyponym_data[i]] - shang_vec[j]
            dks.append(te)
            if j in true_hypernyms[hyponym_data[i]]:
                zheng.append([hyponym_data[i],j,1])
            else:
                fu.append([hyponym_data[i], j, 0])
        cls=xunlina(zheng,fu,top_k)
        test_data =torch.randn(len(dks), n_embeddings * n_clusters)
        predictions = torch.argmax(test_data, dim=1)  # 获取预测类别
        results = []
        for k, (hypo, hyper) in enumerate(dui22):
            score = predictions[k][1]
            results.append([hypo, hyper, score])
        results.sort(key=lambda x: x[2], reverse=True)

        top_hypernym=[]
        for l in results:
            top_hypernym.append(l[1])
        #top_hypernyms = top_hypernym[:kk]
        top_ka.append(top_hypernym)

    return top_ka,top_kb



def jingpai2(a,b, hyponym_data, hypernym_data, true_hypernyms, kk):
    results = []
    hit_count = 0
    true_positive = 0
    reciprocal_ranks = []
    rmse_sum = 0
    ndcg_sum = 0
    dc = 0
    for i in range(len(hyponym_data)):
        top_hypernyms=a[i][:kk]
        top_indices=b[i][:kk]
            # 计算 Hit@k[:kk]
        if len(set(true_hypernyms[hyponym_data[i]]) & set(top_hypernyms)) >= 1:
            hit_count += 1
            true_positive += 1

        dc = dc + len(set(true_hypernyms[hyponym_data[i]]) & set(top_hypernyms))
        for j in top_hypernyms:
            if j in true_hypernyms[hyponym_data[i]]:
                reciprocal_ranks.append(1 / (top_hypernyms.index(j) + 1))

        for j in range(len(true_hypernyms[hyponym_data[i]])):
                # RMSE 计算
            true_relation = 1 if true_hypernyms[hyponym_data[i]][j] in top_hypernyms else 0
            predicted_relation = 1 if top_hypernyms else 0
            rmse_sum += (true_relation - predicted_relation) ** 2

        # NDCG 计算
        dcg = 0
        for j, idx in enumerate(top_indices):
            relevance = 1 if hypernym_data[idx] in true_hypernyms[hyponym_data[i]] else 0
            dcg += relevance / torch.log2(torch.tensor(j + 2.0))  # j + 1 因为索引从0开始
        ideal_dcg = sum(
                [1 / torch.log2(torch.tensor(k + 2.0)) for k in range(len(true_hypernyms[hyponym_data[i]]))])
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
        ndcg_sum += ndcg

        # 收集结果
        results.append((hyponym_data[i], top_hypernyms))

    # 计算评价指标
    total_samples = len(hyponym_data)
    recall = dc / len(true_pairs)
    mrr = sum(reciprocal_ranks) / len(hyponym_data)

    rmse = rmse_sum / total_samples
    ndcg_average = ndcg_sum / total_samples

    hit_at_k = hit_count / total_samples
    print(f'HR: {recall}, Hit@k: {hit_at_k}, MRR: {mrr}, RMSE: {rmse}, NDCG: {ndcg_average}')

    return results, hit_at_k,  recall,  mrr, rmse, ndcg_average

def predict_top_k(model, hyponym_data, hypernym_data, true_hypernyms, top_k):
    model.eval()
    with torch.no_grad():
        x = []
        for i in hyponym_data:
            x.append(xia_vec[i])
        x = torch.tensor(x, dtype=torch.float)
        s = []
        for i in hypernym_data:
            s.append(shang_vec[i])
        s = torch.tensor(s, dtype=torch.float)

        # 前向传播
        scores = model(x, s)
        top_k_scores, top_k_indices = torch.topk(scores, k=top_k, dim=1)


    results = []
    hit_count = 0
    true_positive = 0
    reciprocal_ranks = []
    rmse_sum = 0
    ndcg_sum = 0
    c = 0
    dc=0
    for i in range(len(hyponym_data)):
        top_indices = top_k_indices[i].cpu().numpy()
        top_hypernyms = [hypernym_data[idx] for idx in top_indices]

        b =  len(set(true_hypernyms[hyponym_data[i]]) & set(top_hypernyms))
        # 计算 Hit@k
        if len(set(true_hypernyms[hyponym_data[i]]) & set(top_hypernyms))>=1:
            hit_count += 1
            true_positive += 1

        dc = dc + len(set(true_hypernyms[hyponym_data[i]]) & set(top_hypernyms))
        c = c + b / len(true_hypernyms[hyponym_data[i]])

        for j in range(len(true_hypernyms[hyponym_data[i]])):
            if true_hypernyms[hyponym_data[i]][j] in top_hypernyms:
                reciprocal_ranks.append(1/(top_hypernyms.index(true_hypernyms[hyponym_data[i]][j])+1))
            else:
                reciprocal_ranks.append(0)
            # RMSE 计算
            true_relation = 1 if true_hypernyms[hyponym_data[i]][j] in top_hypernyms else 0
            predicted_relation = 1 if top_hypernyms else 0
            rmse_sum += (true_relation - predicted_relation) ** 2

        # NDCG 计算
        dcg = 0
        for j, idx in enumerate(top_indices):
            relevance = 1 if hypernym_data[idx] in true_hypernyms[hyponym_data[i]] else 0
            dcg += relevance / torch.log2(torch.tensor(j + 2.0))  # j + 1 因为索引从0开始
        ideal_dcg = sum(
            [1 / torch.log2(torch.tensor(k + 2.0)) for k in range(len(true_hypernyms[hyponym_data[i]]))])
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
        ndcg_sum += ndcg

        # 收集结果
        results.append((hyponym_data[i], top_hypernyms))

    # 计算评价指标
    total_samples = len(hyponym_data)
    acc = hit_count / total_samples
    precision = c / len(candidate_hyponyms)
    recall = dc / len(true_pairs)
    f1 = (2 * precision * recall) / (precision + recall)
    map_score = sum([1 / (idx + 1) for idx in range(top_k) if hypernym_data[idx] in true_hypernyms]) / total_samples
    mrr = sum(reciprocal_ranks) / len(true_pairs)

    rmse = rmse_sum / total_samples
    ndcg_average = ndcg_sum / total_samples

    hit_at_k = hit_count / total_samples
    return results, hit_at_k, acc, precision, recall, f1, map_score, mrr, rmse, ndcg_average

def S_X(path):
    # 原始列表格式的 true_pairs
    true_pairs = path

    # 将列表转换为字典格式
    true_dict = {}
    for hyponym, hypernym in true_pairs:
        if hyponym in true_dict:
            true_dict[hyponym].append(hypernym)  # 如果下位词已经存在，追加上位词
        else:
            true_dict[hyponym] = [hypernym]  # 如果下位词不存在，创建新的列表

    return true_dict


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
            ds = []
            d = []
            ddd = []
            dui = []
            # loading...
            warnings.filterwarnings("ignore")
            print('load fast text model...')
            en_model = KeyedVectors.load_word2vec_format(r'')
            print('model load successfully')
            positive_pairs_train = load_pairs('')
            positive_pairs_test = load_pairs('')
            data = ''
            with open("", 'r', encoding="utf-8") as f:
                lines = f.readlines()
                for i in lines[:]:
                    dd = str(i).strip().split(" ")
                    if dd not in ddd:
                        ddd.append(dd)
                    d.append(dd[0].replace(" ", "-"))
                    ds.append(dd[1].replace(" ", "-"))
            candidate_hypernyms = list(set(ds))
            candidate_hyponyms = list(set(d))
            # projection learning
            pos_centroids, pos_new_pairs = cluster_embeddings(positive_pairs_train, en_model)
            print('cluster neg embeddings successfully')
            pos_projections = learn_projections(en_model, pos_new_pairs)
            print('learn pos projections successfully')

            shang_vec = shang(en_model, pos_projections)
            xia_vec = xia(en_model, pos_projections)

            model = train(data, )

            positive_pairs_train = S_X(positive_pairs_test)

            true_pairs = list(set((dd[0], dd[1]) for dd in ddd))
            # 使用示例
            hyponym_data = candidate_hyponyms  # 你的下位词数据
            hypernym_data = candidate_hypernyms  # 你的上位词数据
            true_hypernyms = positive_pairs_train
            a, b = jingpai(model, hyponym_data, hypernym_data, true_hypernyms, 100)
            jingpai2(a, b, hyponym_data, hypernym_data, true_hypernyms, 1)
            jingpai2(a, b, hyponym_data, hypernym_data, true_hypernyms, 3)
            jingpai2(a, b, hyponym_data, hypernym_data, true_hypernyms, 5)
            jingpai2(a, b, hyponym_data, hypernym_data, true_hypernyms, 10)
            jingpai2(a, b, hyponym_data, hypernym_data, true_hypernyms, 15)
            jingpai2(a, b, hyponym_data, hypernym_data, true_hypernyms, 20)

