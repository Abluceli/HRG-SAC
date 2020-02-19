import tensorflow as tf
from .activations import swish, mish
from tensorflow.keras.layers import Dense
from Nn.layers import Noisy, mlp
from GCN.layers import GraphConvolution
from Attention.attention import MultiHeadAttention
import numpy as np
activation_fn = 'tanh'

initKernelAndBias = {
    'kernel_initializer': tf.random_normal_initializer(0.0, .1),
    'bias_initializer': tf.constant_initializer(0.1)    # 2.x 不需要指定dtype
}


class Model(tf.keras.Model):
    def __init__(self, visual_net, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self.visual_net = visual_net
        self.tv = []
        self.tv += self.visual_net.trainable_variables

    def call(self, vector_input, visual_input, *args, **kwargs):
        '''
        args: action, reward, done. etc ...
        '''
        features = self.visual_net(visual_input)

        ret = self.init_or_run(
            tf.concat((vector_input, features), axis=-1),
            *args,
            **kwargs)
        return ret

    def update_vars(self):
        self.tv += self.trainable_variables

    def init_or_run(self, x):
        raise NotImplementedError


class actor_dpg(Model):
    '''
    use for DDPG and/or TD3 algorithms' actor network.
    input: vector of state
    output: deterministic action(mu) and disturbed action(action) given a state
    '''

    def __init__(self, vector_dim, output_shape, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.net = mlp(hidden_units, output_shape=output_shape, out_activation='tanh')

        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim))

        self.update_vars()

    def init_or_run(self, x):
        mu = self.net(x)
        return mu

class actor_dpg_gcn(tf.keras.Model):
    def __init__(self, vector_dim, output_shape, name, hidden_units, visual_net, **kwargs):
        super(actor_dpg_gcn, self).__init__(name=name, **kwargs)
        self.visual_net = visual_net
        self.tv = []
        self.tv += self.visual_net.trainable_variables


        self.gcn_layers = []
        self.gcn_layers.append(GraphConvolution(input_dim=64,
                                                output_dim=64,
                                                num_features_nonzero=0,
                                                activation=tf.nn.relu,
                                                dropout=0.5,
                                                is_sparse_inputs=False,
                                                bias = False,
                                                featureless = False))

        # self.attention_layer = MultiHeadAttention(d_model=64, num_heads=8)
        self.layer_x_embeding = Dense(64, activation='tanh')
        self.layer_a1 = Dense(64, activation='tanh')
        # self.layer_a2 = Dense(64, activation='tanh')

        self.net = Dense(output_shape, activation='tanh')

        self.init_or_run(tf.keras.Input(shape=(vector_dim[0], vector_dim[0])), tf.keras.Input(shape=vector_dim))

        self.update_vars()

    def call(self, adj, x, visual_input):
        features = self.visual_net(visual_input)
        ret = self.init_or_run(adj, x)
        return ret

    def init_or_run(self, adj, x):
        x = self.layer_x_embeding(x)
        outputs = [x]
        for layer in self.gcn_layers:
            hidden = layer((outputs[-1], adj))
            outputs.append(hidden)
        output = outputs[-1]
        # out, attn = self.attention_layer(x, k=x, q=x, mask=None)
        # indices = [[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]]
        # params = [[['a0', 'b0'], ['c0', 'd0']],
        #           [['a1', 'b1'], ['c1', 'd1']]]
        # output = [['b0', 'b1'], ['d0', 'c1']]

        # indices = []
        # if out.shape[0] == None:
        #     for j in range(out.shape[2]):
        #         indices.append([0, j])
        # else:
        #     for i in range(out.shape[0]):
        #         indice = []
        #         for j in range(out.shape[2]):
        #             indice.append([i, 0, j])
        #         indices.append(indice)
        #
        # out = tf.gather_nd(params=out, indices=tf.convert_to_tensor(np.asarray(indices), dtype='int32'), name=None)

        layer_a1 = self.layer_a1(output[:, 0, :])
        # layer_a2 = self.layer_a2(layer_a1)
        ret = self.net(layer_a1)
        return ret

    def update_vars(self):
        self.tv += self.trainable_variables


class actor_mu(Model):
    '''
    use for PPO/PG algorithms' actor network.
    input: vector of state
    output: stochastic action(mu), normally is the mean of a Normal distribution
    '''

    def __init__(self, vector_dim, output_shape, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.net = mlp(hidden_units, output_shape=output_shape, out_activation='tanh')
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim), tf.keras.Input)
        self.update_vars()

    def init_or_run(self, x):
        mu = self.net(x)
        return mu


class actor_continuous(Model):
    '''
    use for continuous action space.
    input: vector of state
    output: mean(mu) and log_variance(log_std) of Gaussian Distribution of actions given a state
    '''

    def __init__(self, vector_dim, output_shape, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.share = mlp(hidden_units['share'], out_layer=False)
        self.mu = mlp(hidden_units['mu'], output_shape=output_shape, out_activation=None)
        self.log_std = mlp(hidden_units['log_std'], output_shape=output_shape, out_activation='tanh')
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim))
        self.update_vars()

    def init_or_run(self, x):
        x = self.share(x)
        mu = self.mu(x)
        log_std = self.log_std(x)
        return (mu, log_std)

class actor_continuous_gcn(tf.keras.Model):
    def __init__(self, vector_dim, output_shape, name, hidden_units, visual_net, **kwargs):
        super(actor_continuous_gcn, self).__init__(name=name, **kwargs)
        self.visual_net = visual_net
        self.tv = []
        self.tv += self.visual_net.trainable_variables

        self.gcn_layers1 = GraphConvolution(input_dim=64,
                                                output_dim=64,
                                                num_features_nonzero=0,
                                                activation=tf.nn.relu,
                                                dropout=0.5,
                                                is_sparse_inputs=False,
                                                bias=False,
                                                featureless=False)
        # self.gcn_layers2 = GraphConvolution(input_dim=64,
        #                                     output_dim=64,
        #                                     num_features_nonzero=0,
        #                                     activation=tf.nn.relu,
        #                                     dropout=0.5,
        #                                     is_sparse_inputs=False,
        #                                     bias=False,
        #                                     featureless=False)
        #
        # self.attention_layer1 = MultiHeadAttention(d_model=64, num_heads=8)
        # self.attention_layer2 = MultiHeadAttention(d_model=64, num_heads=8)
        self.layer_x_embeding = Dense(64, activation='tanh')

        # self.share = mlp(hidden_units['share'], out_layer=False)
        self.layer_1 = Dense(64, activation='tanh')
        self.mu = mlp(hidden_units['mu'], output_shape=output_shape, out_activation=None)
        self.log_std = mlp(hidden_units['log_std'], output_shape=output_shape, out_activation='tanh')

        self.init_or_run(tf.keras.Input(shape=(vector_dim[0], vector_dim[0])), tf.keras.Input(shape=vector_dim))
        self.update_vars()


    def call(self, adj, x, visual_input):
        features = self.visual_net(visual_input)
        mu, log_std = self.init_or_run(adj, x)
        return (mu, log_std)

    def init_or_run(self, adj, x):
        x = self.layer_x_embeding(x)

        hidden1 = self.gcn_layers1((x, adj))
        # out1, attn1 = self.attention_layer1(hidden1, k=hidden1, q=hidden1, mask=None)
        #
        # hidden2 = self.gcn_layers1((out1, adj))
        # out2, attn2 = self.attention_layer2(hidden2, k=hidden2, q=hidden2, mask=None)

        layer_1 = self.layer_1(tf.concat((x[:, 0, :], hidden1[:, 0, :]), axis=-1))
        # layer_1 = self.layer_1(x[:, 0, :])
        mu = self.mu(layer_1)
        log_std = self.log_std(layer_1)
        return (mu, log_std)
    def update_vars(self):
        self.tv += self.trainable_variables



class actor_discrete(Model):
    '''
    use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state
    '''

    def __init__(self, vector_dim, output_shape, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.logits = mlp(hidden_units, output_shape=output_shape, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim))
        self.update_vars()

    def init_or_run(self, x):
        logits = self.logits(x)
        return logits


class critic_q_one(Model):
    '''
    use for evaluate the value given a state-action pair.
    input: tf.concat((state, action),axis = 1)
    output: q(s,a)
    '''

    def __init__(self, vector_dim, action_dim, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.net = mlp(hidden_units, output_shape=1, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim), tf.keras.Input(shape=action_dim))
        self.update_vars()

    def init_or_run(self, x, a):

        q = self.net(tf.concat((x, a), axis=-1))
        return q


class critic_q_one_gcn(tf.keras.Model):
    def __init__(self, vector_dim, output_shape, name, hidden_units, visual_net, **kwargs):
        super(critic_q_one_gcn, self).__init__(name=name, **kwargs)
        self.visual_net = visual_net
        self.tv = []
        self.tv += self.visual_net.trainable_variables

        # self.gcn_layers = []
        # self.gcn_layers.append(GraphConvolution(input_dim=128,
        #                                         output_dim=128,
        #                                         num_features_nonzero=0,
        #                                         activation=tf.nn.relu,
        #                                         dropout=0.5,
        #                                         is_sparse_inputs=False))

        # self.attention_layer = MultiHeadAttention(d_model=64, num_heads=8)
        self.layer_x_embeding = Dense(64, activation='tanh')
        self.layer_c1 = Dense(32, activation='tanh')
        self.layer_c2 = Dense(64, activation='tanh')
        self.value = Dense(1)
        self.init_or_run(tf.keras.Input(shape=(vector_dim[0], vector_dim[0])), tf.keras.Input(shape=vector_dim), tf.keras.Input(shape=output_shape))
        self.update_vars()

    def call(self, adj, x, visual_input, a):
        features = self.visual_net(visual_input)
        ret = self.init_or_run(adj, x, a)
        return ret

    def init_or_run(self, adj, x, a):
        x = self.layer_x_embeding(x)
        # outputs = [x]
        # for layer in self.gcn_layers:
        #     hidden = layer((outputs[-1], adj))
        #     outputs.append(hidden)
        # output = outputs[-1]

        # out, attn = self.attention_layer(x, k=x, q=x, mask=None)

        action_emb = self.layer_c1(a)
        state_action = tf.concat([x[:, 0, :], action_emb], -1)
        layer_c2 = self.layer_c2(state_action)
        value = self.value(layer_c2)
        return value

    def update_vars(self):
        self.tv += self.trainable_variables


class critic_q_one2(Model):
    '''
    Original architecture in DDPG paper.
    s-> layer -> feature, then tf.concat(feature, a) -> layer -> output
    '''

    def __init__(self, vector_dim, action_dim, name, hidden_units, *, visual_net):
        assert len(hidden_units) > 1, "if you want to use this architecture of critic network, the number of layers must greater than 1"
        super().__init__(visual_net, name=name)
        self.state_feature_net = mlp(hidden_units[0:1])
        self.net = mlp(hidden_units[1:], output_shape=1, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim), tf.keras.Input(shape=action_dim))
        self.update_vars()

    def init_or_run(self, x, a):
        features = self.state_feature_net(x)
        q = self.net(tf.concat((x, a), axis=-1))
        return q


class critic_q_one3(Model):
    '''
    Original architecture in TD3 paper.
    tf.concat(s,a) -> layer -> feature, then tf.concat(feature, a) -> layer -> output
    '''

    def __init__(self, vector_dim, action_dim, name, hidden_units, *, visual_net):
        assert len(hidden_units) > 1, "if you want to use this architecture of critic network, the number of layers must greater than 1"
        super().__init__(visual_net, name=name)
        self.feature_net = mlp(hidden_units[0:1])
        self.net = mlp(hidden_units[1:], output_shape=1, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim), tf.keras.Input(shape=action_dim))
        self.update_vars()

    def init_or_run(self, x, a):
        features = self.feature_net(tf.concat((x, a), axis=-1))
        q = self.net(tf.concat((features, a), axis=-1))
        return q


class critic_v(Model):
    '''
    use for evaluate the value given a state.
    input: vector of state
    output: v(s)
    '''

    def __init__(self, vector_dim, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.net = mlp(hidden_units, output_shape=1, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim))
        self.update_vars()

    def init_or_run(self, x):
        v = self.net(x)
        return v


class critic_q_all(Model):
    '''
    use for evaluate all values of Q(S,A) given a state. must be discrete action space.
    input: vector of state
    output: q(s, *)
    '''

    def __init__(self, vector_dim, output_shape, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.net = mlp(hidden_units, output_shape=output_shape, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim))
        self.update_vars()

    def init_or_run(self, x):
        q = self.net(x)
        return q

class critic_q_all_gcn(tf.keras.Model):
    def __init__(self, vector_dim, output_shape, name, hidden_units, visual_net, **kwargs):
        super(critic_q_all_gcn, self).__init__(name=name, **kwargs)
        self.visual_net = visual_net
        self.tv = []
        self.tv += self.visual_net.trainable_variables

        # self.gcn_layers = []
        # self.gcn_layers.append(GraphConvolution(input_dim=64,
        #                                         output_dim=64,
        #                                         num_features_nonzero=0,
        #                                         activation=tf.nn.relu,
        #                                         dropout=0.5,
        #                                         is_sparse_inputs=False))

        # self.attention_layer1 = MultiHeadAttention(d_model=64, num_heads=8)
        # self.attention_layer2 = MultiHeadAttention(d_model=64, num_heads=8)
        self.layer_x_embeding = Dense(64, activation='tanh')
        self.layer_c1 = Dense(64, activation='tanh')
        # self.layer_c2 = Dense(64, activation='tanh')
        self.value = Dense(output_shape)
        self.init_or_run(tf.keras.Input(shape=(vector_dim[0], vector_dim[0])), tf.keras.Input(shape=vector_dim))
        self.update_vars()

    def call(self, adj, x, visual_input):
        features = self.visual_net(visual_input)
        value = self.init_or_run(adj, x)
        return value

    def init_or_run(self, adj, x):
        x = self.layer_x_embeding(x)
        # outputs = [x]
        # for layer in self.gcn_layers:
        #     hidden = layer((outputs[-1], adj))
        #     outputs.append(hidden)
        # output = outputs[-1]
        # out1, attn1 = self.attention_layer1(x, k=x, q=x, mask=None)
        # out2, attn2 = self.attention_layer2(out1, k=out1, q=out1, mask=None)
        layer_c1 = self.layer_c1(x[:, 0, :])
        value = self.value(layer_c1)
        return value

    def update_vars(self):
        self.tv += self.trainable_variables



class drqn_critic_q_all(Model):
    '''
    use for evaluate all values of Q(S,A) given a state. must be discrete action space.
    input: vector of state
    output: q(s, *)
    '''

    def __init__(self, vector_dim, output_shape, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.masking = tf.keras.layers.Masking(mask_value=0.)
        self.lstm_net = tf.keras.layers.LSTM(hidden_units['lstm'], return_state=True, return_sequences=True)
        self.net = mlp(hidden_units['dense'], output_shape=output_shape, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=(None, vector_dim + self.visual_net.hdim)))
        self.update_vars()

    def init_or_run(self, x, initial_state=None):
        x = self.masking(x)
        if initial_state is not None:
            x, h, c = self.lstm_net(x, initial_state=initial_state)
        else:
            x, h, c = self.lstm_net(x)
        q = self.net(x)
        q = tf.reshape(q, (-1, q.shape[-1]))    # [B, T, 1] => [B*T, 1]
        return (q, [h, c])


class critic_dueling(Model):
    '''
    Neural network for dueling deep Q network.
    Input:
        states: [batch_size, state_dim]
    Output:
        state value: [batch_size, 1]
        advantage: [batch_size, action_number]
    '''

    def __init__(self, vector_dim, output_shape, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.share = mlp(hidden_units['share'], out_layer=False)
        self.v = mlp(hidden_units['v'], output_shape=1, out_activation=None)
        self.adv = mlp(hidden_units['adv'], output_shape=output_shape, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim))
        self.update_vars()

    def init_or_run(self, x):
        x = self.share(x)
        v = self.v(x)    # [B, 1]
        adv = self.adv(x)  # [B, A]
        q = v + adv - tf.reduce_mean(adv, axis=1, keepdims=True)  # [B, A]
        return q


class a_c_v_continuous(Model):
    '''
    combine actor network and critic network, share some nn layers. use for continuous action space.
    input: vector of state
    output: mean(mu) of Gaussian Distribution of actions given a state, v(s)
    '''

    def __init__(self, vector_dim, output_shape, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.share = mlp(hidden_units['share'], out_layer=False)
        self.mu = mlp(hidden_units['mu'], output_shape=output_shape, out_activation='tanh')
        self.v = mlp(hidden_units['v'], output_shape=1, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim))
        self.update_vars()

    def init_or_run(self, x):
        x = self.share(x)
        mu = self.mu(x)
        v = self.v(x)
        return (mu, v)


class a_c_v_discrete(Model):
    '''
    combine actor network and critic network, share some nn layers. use for discrete action space.
    input: vector of state
    output: probability distribution of actions given a state, v(s)
    '''

    def __init__(self, vector_dim, output_shape, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.share = mlp(hidden_units['share'], out_layer=False)
        self.logits = mlp(hidden_units['logits'], output_shape=output_shape, out_activation=None)
        self.v = mlp(hidden_units['v'], output_shape=1, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim))
        self.update_vars()

    def init_or_run(self, x):
        x = self.share(x)
        logits = self.logits(x)
        v = self.v(x)
        return (logits, v)


class c51_distributional(Model):
    '''
    neural network for C51
    '''

    def __init__(self, vector_dim, action_dim, atoms, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.action_dim = action_dim
        self.atoms = atoms
        self.net = mlp(hidden_units, output_shape=atoms * action_dim, out_activation='softmax')
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim))
        self.update_vars()

    def init_or_run(self, x):
        q_dist = self.net(x)    # [B, A*N]
        q_dist = tf.reshape(q_dist, [-1, self.action_dim, self.atoms])   # [B, A, N]
        return q_dist

class qrdqn_distributional(Model):
    '''
    neural network for QRDQN
    '''

    def __init__(self, vector_dim, action_dim, nums, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.action_dim = action_dim
        self.nums = nums
        self.net = mlp(hidden_units, output_shape=nums * action_dim, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim))
        self.update_vars()

    def init_or_run(self, x):
        q_dist = self.net(x)    # [B, A*N]
        q_dist = tf.reshape(q_dist, [-1, self.action_dim, self.nums])   # [B, A, N]
        return q_dist


class rainbow_dueling(Model):
    '''
    Neural network for Rainbow.
    Input:
        states: [batch_size, state_dim]
    Output:
        state value: [batch_size, atoms]
        advantage: [batch_size, action_number * atoms]
    '''

    def __init__(self, vector_dim, action_dim, atoms, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.action_dim = action_dim
        self.atoms = atoms
        self.share = mlp(hidden_units['share'], layer=Noisy, out_layer=False)
        self.v = mlp(hidden_units['v'], layer=Noisy, output_shape=atoms, out_activation=None)
        self.adv = mlp(hidden_units['adv'], layer=Noisy, output_shape=action_dim * atoms, out_activation=None)
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim))
        self.update_vars()

    def init_or_run(self, x):
        x = self.share(x)
        v = self.v(x)    # [B, N]
        adv = self.adv(x)   # [B, A*N]
        adv = tf.reshape(adv, [-1, self.action_dim, self.atoms])   # [B, A, N]
        adv -= tf.reduce_mean(adv)  # [B, A, N]
        adv = tf.transpose(adv, [1, 0, 2])  # [A, B, N]
        q = tf.transpose(v + adv, [1, 0, 2])    # [B, A, N]
        q = tf.nn.softmax(q)    # [B, A, N]
        return q  # [B, A, N]


class iqn_net(Model):
    def __init__(self, vector_dim, action_dim, quantiles_idx, name, hidden_units, *, visual_net):
        super().__init__(visual_net, name=name)
        self.action_dim = action_dim
        self.q_net_head = mlp(hidden_units['q_net'], out_layer=False)   # [B, vector_dim]
        self.quantile_net = mlp(hidden_units['quantile'], out_layer=False)  # [N*B, quantiles_idx]
        self.q_net_tile = mlp(hidden_units['tile'], output_shape=action_dim, out_activation=None)   # [N*B, hidden_units['quantile'][-1]]
        self.init_or_run(tf.keras.Input(shape=vector_dim + self.visual_net.hdim), tf.keras.Input(shape=quantiles_idx))
        self.update_vars()

    def init_or_run(self, x, quantiles_tiled, *, quantiles_num=8):
        q_h = self.q_net_head(x)  # [B, obs_dim] => [B, h]
        q_h = tf.tile(q_h, [quantiles_num, 1])  # [B, h] => [N*B, h]
        quantile_h = self.quantile_net(quantiles_tiled)  # [N*B, quantiles_idx] => [N*B, h]
        hh = q_h * quantile_h  # [N*B, h]
        quantiles_value = self.q_net_tile(hh)  # [N*B, h] => [N*B, A]
        quantiles_value = tf.reshape(quantiles_value, (quantiles_num, -1, self.action_dim))   # [N*B, A] => [N, B, A]
        q = tf.reduce_mean(quantiles_value, axis=0)  # [N, B, A] => [B, A]
        return (quantiles_value, q)
