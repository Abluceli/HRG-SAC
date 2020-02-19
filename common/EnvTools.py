from mlagents.envs.environment import UnityEnvironment
import numpy as np
import scipy.sparse as sp

def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """
    Row-normalize feature matrix and convert to tuple representation
    """
    rowsum = np.array(features.sum(1)) # get sum of each row, [2708, 1]
    r_inv = np.power(rowsum, -1).flatten() # 1/rowsum, [2708]
    r_inv[np.isinf(r_inv)] = 0. # zero inf data
    r_mat_inv = sp.diags(r_inv) # sparse diagonal matrix, [2708, 2708]
    features = r_mat_inv.dot(features) # D^-1:[2708, 2708]@X:[2708, 2708]
    return features # [coordinates, data, shape], []


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^0.5


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized.A

def preprocess_observation_no_n(obs):

    #obs: 9+12x5
    NODE_NUM = 13
    X_WEIDU = 9
    NODE_WEIDU = 5
    X = []
    A = []
    for _, ob in enumerate(obs):
        A_ = np.zeros((NODE_NUM, NODE_NUM))
        for i in range(NODE_NUM):
            if i!=0:
                A_[0][i] = 1
                A_[i][0] = 1
        X_ = np.zeros((NODE_NUM, X_WEIDU))

        for i in range(NODE_NUM):
            if i==0:
                X_[i] = ob[0 : X_WEIDU]
            else:
                for j in range(NODE_WEIDU):
                    X_[i][j] = ob[X_WEIDU + (i - 1) * NODE_WEIDU + j]
        A.append(preprocess_adj(A_))
        X.append(X_)
    return np.array(A), np.array(X)

def preprocess_observation_n(obs):
    """
    for each agent in unity env, #为每一个agent找到与他最近的一个agent，
    和与两个agent最近的n个food表示为（1+1+n）x（1+1+n）
    维度的邻接矩阵 A，和（1+1+n）x l的特征矩阵 X
    return A, X
    """
    #obs: 1x6+49+(30+30)x6
    NUM_FOOD = 30
    NUM_AGENT = 5
    X_WEIDU = 41
    NODE_WEIDU = 6
    n = 16
    agentInfos = np.zeros((NUM_AGENT, X_WEIDU))
    foodInfos = np.zeros((NUM_FOOD, X_WEIDU))
    obs_ = np.zeros((NUM_AGENT+NUM_FOOD, X_WEIDU))
    for i, o in enumerate(obs):
        agentInfos[i] = o[0: X_WEIDU]
        obs_[i] = o[0: X_WEIDU]
    fInfos = obs[0][X_WEIDU:]
    for i in range(int(len(fInfos)/NODE_WEIDU)):
        obs_[i+NUM_AGENT][:NODE_WEIDU] = fInfos[i * NODE_WEIDU:(i + 1) * NODE_WEIDU]
        if i < NUM_FOOD:
            foodInfos[i][:NODE_WEIDU] = fInfos[i * NODE_WEIDU:(i + 1) * NODE_WEIDU]

    # for o in obs_:
    #     print(o)
    # print(len(obs_), len(agentInfos), len(foodInfos), len(badFoodsInfos))


    A_agent = []
    # dis = []

    for j in range(len(agentInfos)):
        f = []
        for r in range(len(agentInfos)):
            f.append([(agentInfos[r][0] - agentInfos[j][0]) ** 2 + (agentInfos[r][1] - agentInfos[j][1]) ** 2, r])
        f.sort(key=lambda x: x[0])
        A_agent.append([f[0][1], f[1][1]])
    # print(A_agent) #[[0, 2], [1, 2], [2, 1], [3, 4], [4, 1]]



    A_agent_food = []
    for a in A_agent:
        f = []
        for r in range(len(foodInfos)):
            f.append([
                (foodInfos[r][0] - agentInfos[a[0]][0]) ** 2 + (foodInfos[r][1] - agentInfos[a[0]][1]) ** 2 +
                (foodInfos[r][0] - agentInfos[a[1]][0]) ** 2 + (foodInfos[r][1] - agentInfos[a[1]][1]) ** 2
                , r
            ])
        f.sort(key=lambda x: x[0])
        A_agent_food.append([f[i][1] for i in range(n)])
    #print(A_agent_food)#[[33, 41, 16, 31], [22, 37, 31, 39], [22, 37, 31, 39], [20, 0, 8, 1], [26, 46, 10, 47]]

    A_agent_food_ = []
    for i, af in enumerate(A_agent_food):
        a = [[], []]
        for f in af:
            if ((foodInfos[f][0] - agentInfos[A_agent[i][0]][0]) ** 2 + (foodInfos[f][1] - agentInfos[A_agent[i][0]][1]) ** 2) - \
                    ((foodInfos[f][0] - agentInfos[A_agent[i][1]][0]) ** 2 + (foodInfos[f][1] - agentInfos[A_agent[i][1]][1]) ** 2) < 0:
                a[0].append(f)
            else:
                a[1].append(f)
        A_agent_food_.append(a)
    # print(A_agent_food_) #[[[33, 41], [16, 31]], [[22, 37, 31], [39]], [[39], [22, 37, 31]], [[20], [0, 8, 1]], [[26, 46, 10, 47], []]]

    X = []
    A = []
    for i in range(len(A_agent)):
        index = []
        index_relation = []
        agent = A_agent[i]
        a = A_agent_food_[i]
        for j in agent:
            index.append(j)
        for j in a:
            if len(j) > 0:
                for s in j:
                    index.append(s+5)
        index_relation.append(agent)
        for j in range(len(agent)):
            if len(a[j]) > 0:
                for s in a[j]:
                    index_relation.append([agent[j], s+5])
        A_ = np.zeros(shape=(len(index), len(index)))
        X_ = []
        for relation in index_relation:
            A_[index.index(relation[0])][index.index(relation[1])] = 1
            A_[index.index(relation[1])][index.index(relation[0])] = 1
        for ind in index:
            X_.append(obs_[ind])
        A.append(preprocess_adj(A_))
        X.append(X_)
    return np.array(A), np.array(X)

def preprocess_observation_2n(obs):
    """
    for each agent in unity env, #为每一个agent找到与他最近的一个agent，
    和与两个agent最近的n个food和n个 badfood 表示为（1+1+n+n）x（1+1+n+n）
    维度的邻接矩阵 A，和（1+1+n+n）x l的特征矩阵 X
    return A, X
    """
    #obs: 1x6+49+(30+30)x6
    NUM_FOOD = 30
    NUM_BADFOOD = 30
    NUM_AGENT = 5
    X_WEIDU = 58
    agentInfos = np.zeros((NUM_AGENT, X_WEIDU))
    foodInfos = np.zeros((NUM_FOOD, X_WEIDU))
    badFoodsInfos = np.zeros((NUM_BADFOOD, X_WEIDU))
    obs_ = np.zeros((NUM_AGENT+NUM_FOOD+NUM_BADFOOD, X_WEIDU))
    for i, o in enumerate(obs):
        agentInfos[i] = o[0: X_WEIDU]
        obs_[i] = o[0: X_WEIDU]
    fInfos = obs[0][X_WEIDU:]
    for i in range(int(len(fInfos)/X_WEIDU)):
        obs_[i+NUM_AGENT] = fInfos[i * X_WEIDU:(i + 1) * X_WEIDU]
        if i < NUM_FOOD:
            foodInfos[i] = fInfos[i * X_WEIDU:(i + 1) * X_WEIDU]
        elif i < NUM_FOOD + NUM_BADFOOD:
            badFoodsInfos[i-NUM_FOOD] = fInfos[i * X_WEIDU:(i + 1) * X_WEIDU]
    # for o in obs_:
    #     print(o)
    # print(len(obs_), len(agentInfos), len(foodInfos), len(badFoodsInfos))


    A_agent = []
    # dis = []

    for j in range(len(agentInfos)):
        f = []
        for r in range(len(agentInfos)):
            f.append([(agentInfos[r][0] - agentInfos[j][0]) ** 2 + (agentInfos[r][1] - agentInfos[j][1]) ** 2, r])
        f.sort(key=lambda x: x[0])
        A_agent.append([f[0][1], f[1][1]])
    # print(A_agent) #[[0, 2], [1, 2], [2, 1], [3, 4], [4, 1]]


    n = 4
    A_agent_food = []
    for a in A_agent:
        f = []
        for r in range(len(foodInfos)):
            f.append([
                (foodInfos[r][0] - agentInfos[a[0]][0]) ** 2 + (foodInfos[r][1] - agentInfos[a[0]][1]) ** 2 +
                (foodInfos[r][0] - agentInfos[a[1]][0]) ** 2 + (foodInfos[r][1] - agentInfos[a[1]][1]) ** 2
                , r
            ])
        f.sort(key=lambda x: x[0])
        A_agent_food.append([f[i][1] for i in range(n)])
    #print(A_agent_food)#[[33, 41, 16, 31], [22, 37, 31, 39], [22, 37, 31, 39], [20, 0, 8, 1], [26, 46, 10, 47]]

    A_agent_food_ = []
    for i, af in enumerate(A_agent_food):
        a = [[], []]
        for f in af:
            if ((foodInfos[f][0] - agentInfos[A_agent[i][0]][0]) ** 2 + (foodInfos[f][1] - agentInfos[A_agent[i][0]][1]) ** 2) - ((foodInfos[f][0] - agentInfos[A_agent[i][1]][0]) ** 2 + (foodInfos[f][1] - agentInfos[A_agent[i][1]][1]) ** 2) < 0:
                a[0].append(f)
            else:
                a[1].append(f)
        A_agent_food_.append(a)
    # print(A_agent_food_) #[[[33, 41], [16, 31]], [[22, 37, 31], [39]], [[39], [22, 37, 31]], [[20], [0, 8, 1]], [[26, 46, 10, 47], []]]


    A_agent_badfood = []
    for a in A_agent:
        f = []
        for r in range(len(badFoodsInfos)):
            f.append([
                (badFoodsInfos[r][0] - agentInfos[a[0]][0]) ** 2 + (badFoodsInfos[r][1] - agentInfos[a[0]][1]) ** 2 +
                (badFoodsInfos[r][0] - agentInfos[a[1]][0]) ** 2 + (badFoodsInfos[r][1] - agentInfos[a[1]][1]) ** 2
                , r
            ])
        f.sort(key=lambda x: x[0])
        A_agent_badfood.append([f[i][1] for i in range(n)])
    # print(A_agent_badfood)#[[33, 41, 16, 31], [22, 37, 31, 39], [22, 37, 31, 39], [20, 0, 8, 1], [26, 46, 10, 47]]

    A_agent_badfood_ = []
    for i, af in enumerate(A_agent_badfood):
        a = [[], []]
        for f in af:
            if ((badFoodsInfos[f][0] - agentInfos[A_agent[i][0]][0]) ** 2 + (badFoodsInfos[f][1] - agentInfos[A_agent[i][0]][1]) ** 2) - ((badFoodsInfos[f][0] - agentInfos[A_agent[i][1]][0]) ** 2 + (badFoodsInfos[f][1] - agentInfos[A_agent[i][1]][1]) ** 2) < 0:
                a[0].append(f)
            else:
                a[1].append(f)
        A_agent_badfood_.append(a)
    # print(A_agent_badfood_) #[[[33, 41], [16, 31]], [[22, 37, 31], [39]], [[39], [22, 37, 31]], [[20], [0, 8, 1]], [[26, 46, 10, 47], []]]

    X = []
    A = []
    for i in range(len(A_agent)):
        index = []
        index_relation = []
        agent = A_agent[i]
        a = A_agent_food_[i]
        b = A_agent_badfood_[i]
        for j in agent:
            index.append(j)
        for j in a:
            if len(j) > 0:
                for s in j:
                    index.append(s+5)
        for j in b:
            if len(j) > 0:
                for s in j:
                    index.append(s+25)
        index_relation.append(agent)
        for j in range(len(agent)):
            if len(a[j]) > 0:
                for s in a[j]:
                    index_relation.append([agent[j], s+5])
            if len(b[j]) > 0:
                for s in b[j]:
                    index_relation.append([agent[j], s+25])

        A_ = np.zeros(shape=(len(index), len(index)))
        X_ = []
        for relation in index_relation:
            A_[index.index(relation[0])][index.index(relation[1])] = 1
            A_[index.index(relation[1])][index.index(relation[0])] = 1
        for ind in index:
            X_.append(obs_[ind])
        A.append(preprocess_adj(A_))
        X.append(X_)
    return np.array(A), np.array(X)

def preprocess_observation_4n(obs):
    """
    for each agent in unity env, #为每一个agent找到与他最近的一个agent，
    ，再分别找与每个agent最近的n个food和n个 badfood， 表示为（1+1+n+n+n+n）x（1+1+n+n+n+n）
    维度的邻接矩阵 A，和（1+1+n+n+n+n）x l的特征矩阵 X
    return A, X
    """
    #obs: (1x6+49+(30+30)x6)x5
    n = 4
    NUM_FOOD = 30
    NUM_BADFOOD = 30
    NUM_AGENT = 5
    X_WEIDU = 55
    agentInfos = np.zeros((NUM_AGENT, X_WEIDU))
    foodInfos = np.zeros((NUM_FOOD, X_WEIDU))
    badFoodsInfos = np.zeros((NUM_BADFOOD, X_WEIDU))
    # obs_ = np.zeros((NUM_AGENT+NUM_FOOD+NUM_BADFOOD, X_WEIDU))
    for i, o in enumerate(obs):
        agentInfos[i] = o[0: X_WEIDU]
        # obs_[i] = o[0: X_WEIDU]
    fInfos = obs[0][X_WEIDU:]
    for i in range(int(len(fInfos)/6)):
        # obs_[i+NUM_AGENT][:6] = fInfos[i * 6:(i + 1) * 6]
        if i < NUM_FOOD:
            foodInfos[i][:6] = fInfos[i * 6:(i + 1) * 6]
        elif i < NUM_FOOD + NUM_BADFOOD:
            badFoodsInfos[i-NUM_FOOD][:6] = fInfos[i * 6:(i + 1) * 6]
    # for o in obs_:
    #     print(o)
    # print(len(obs_), len(agentInfos), len(foodInfos), len(badFoodsInfos))


    A_agent = []
    # dis = []

    for j in range(len(agentInfos)):
        f = []
        for r in range(len(agentInfos)):
            f.append([(agentInfos[r][0] - agentInfos[j][0]) ** 2 + (agentInfos[r][1] - agentInfos[j][1]) ** 2, r])
        f.sort(key=lambda x: x[0])
        A_agent.append([f[0][1], f[1][1]])
    # print(A_agent) #[[0, 2], [1, 2], [2, 1], [3, 4], [4, 1]]



    A_agent_food = []
    for a in A_agent:
        f = []
        for r in range(len(foodInfos)):
            f.append([
                (foodInfos[r][0] - agentInfos[a[0]][0]) ** 2 + (foodInfos[r][1] - agentInfos[a[0]][1]) ** 2
                , r
            ])
        f.sort(key=lambda x: x[0])
        f_ = []
        for r in range(len(foodInfos)):
            f_.append([
                (foodInfos[r][0] - agentInfos[a[1]][0]) ** 2 + (foodInfos[r][1] - agentInfos[a[1]][1]) ** 2
                , r
            ])
        f_.sort(key=lambda x: x[0])
        A_agent_food.append([f[i][1] for i in range(n)]+[f_[i][1] for i in range(n)])
        # for i in range(NUM_AGENT):
        #     for j in range(n):
        #         A_agent_food[i].append(f_[j][1])
    # print(A_agent_food)#[[33, 41, 16, 31], [22, 37, 31, 39], [22, 37, 31, 39], [20, 0, 8, 1], [26, 46, 10, 47]]


    A_agent_badfood = []
    for a in A_agent:
        f = []
        for r in range(len(badFoodsInfos)):
            f.append([
                (badFoodsInfos[r][0] - agentInfos[a[0]][0]) ** 2 + (badFoodsInfos[r][1] - agentInfos[a[0]][1]) ** 2
                , r
            ])
        f.sort(key=lambda x: x[0])
        f_ = []
        for r in range(len(badFoodsInfos)):
            f_.append([
                (badFoodsInfos[r][0] - agentInfos[a[1]][0]) ** 2 + (badFoodsInfos[r][1] - agentInfos[a[1]][1]) ** 2
                , r
            ])
        f_.sort(key=lambda x: x[0])

        A_agent_badfood.append([f[i][1] for i in range(n)] + [f_[i][1] for i in range(n)])
        # for i in range(NUM_AGENT):
        #     for j in range(n):
        #         A_agent_badfood[i].append(f_[j][1])
    # print(A_agent_badfood)#[[33, 41, 16, 31], [22, 37, 31, 39], [22, 37, 31, 39], [20, 0, 8, 1], [26, 46, 10, 47]]

    X = []

    for i in range(len(A_agent)):
        x_ = np.zeros((1+1+n+n+n+n, X_WEIDU))
        j = 0
        for agentid in A_agent[i]:
            x_[j] = agentInfos[agentid]
            j = j + 1
        for foodid in A_agent_food[i]:
            x_[j] = foodInfos[foodid]
            j = j + 1
        for badfoodid in A_agent_badfood[i]:
            x_[j] = badFoodsInfos[badfoodid]
            j = j + 1
        X.append(x_)

    A = []

    A_ = np.zeros(shape=(1+1+n+n+n+n, 1+1+n+n+n+n))
    for i in range(1+1):
        for j in range(1+1+n+n+n+n):
            if i==0:
                if j==1:
                    A_[i][j] = 1
                    A_[j][i] = 1
                if 1<j and j< 1+1+n:
                    A_[i][j] = 1
                    A_[j][i] = 1
                if 1 + n + n < j and j < 1 + 1 + n + n + n:
                    A_[i][j] = 1
                    A_[j][i] = 1
            if i==1:
                if j == 0:
                    A_[i][j] = 1
                    A_[j][i] = 1
                if 1 + n < j and j < 1 + 1 + n + n:
                    A_[i][j] = 1
                    A_[j][i] = 1
                if 1 + n + n + n < j and j < 1 + 1 + n + n + n + n:
                    A_[i][j] = 1
                    A_[j][i] = 1
    for i in range(1+1+n+n+n+n):
        A_[i][i] = 1
    for a in range(NUM_AGENT):
        A.append(preprocess_adj(A_))


    return np.array(A), np.array(X)



if __name__ == '__main__':
    env = UnityEnvironment()
    obs = env.reset(train_mode=True)
    brain_name = env.brain_names[0]
    obs = obs[brain_name].vector_observations
    a , x = preprocess_observation_n(obs)
    print(a)
    print(x)
    env.close()

