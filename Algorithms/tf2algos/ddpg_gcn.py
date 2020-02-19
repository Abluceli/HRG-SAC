import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import Nn
from utils.sth import sth
from Algorithms.tf2algos.base.off_policy import Off_Policy


class DDPG_GCN(Off_Policy):
    '''
    Deep Deterministic Policy Gradient, https://arxiv.org/abs/1509.02971
    '''
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 is_continuous,

                 ployak=0.995,
                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 share_visual_net=True,
                 discrete_tau=1.0,
                 hidden_units={
                     'actor_continuous': [64, 64],
                     'actor_discrete': [64, 64],
                     'q': [64, 64]
                 },

                 **kwargs):
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim_or_list=a_dim_or_list,
            is_continuous=is_continuous,
            **kwargs)
        self.ployak = ployak
        self.discrete_tau = discrete_tau

        self.share_visual_net = share_visual_net
        if self.share_visual_net:
            self.actor_visual_net = self.critic_visual_net = Nn.VisualNet('visual_net', self.visual_dim)
        else:
            self.actor_visual_net = Nn.VisualNet('actor_visual_net', self.visual_dim)
            self.critic_visual_net = Nn.VisualNet('critic_visual_net', self.visual_dim)

        if self.is_continuous:
            self.actor_net = Nn.actor_dpg_gcn(vector_dim=self.s_dim, output_shape=self.a_counts, name='actor_net', hidden_units=hidden_units['actor_continuous'], visual_net=self.actor_visual_net)
            self.actor_target_net = Nn.actor_dpg_gcn(vector_dim=self.s_dim, output_shape=self.a_counts, name='actor_net', hidden_units=hidden_units['actor_continuous'], visual_net=self.actor_visual_net)
            # self.action_noise = Nn.NormalActionNoise(mu=np.zeros(self.a_counts), sigma=1 * np.ones(self.a_counts))
            self.action_noise = Nn.OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.a_counts), sigma=0.5 * np.exp(-self.episode / 10) * np.ones(self.a_counts))
        else:
            self.actor_net = Nn.actor_discrete(self.s_dim, self.a_counts, 'actor_net', hidden_units['actor_discrete'], visual_net=self.actor_visual_net)
            self.actor_target_net = Nn.actor_discrete(self.s_dim, self.a_counts, 'actor_target_net', hidden_units['actor_discrete'], visual_net=self.actor_visual_net)
            self.gumbel_dist = tfp.distributions.Gumbel(0, 1)
        self.q_net = Nn.critic_q_one_gcn(vector_dim=self.s_dim, output_shape=self.a_counts, name='q_net', hidden_units=hidden_units['q'], visual_net=self.critic_visual_net)
        self.q_target_net = Nn.critic_q_one_gcn(vector_dim=self.s_dim, output_shape=self.a_counts, name='q_net', hidden_units=hidden_units['q'], visual_net=self.critic_visual_net)
        self.update_target_net_weights(
            self.actor_target_net.weights + self.q_target_net.weights,
            self.actor_net.weights + self.q_net.weights
        )
        self.actor_lr = tf.keras.optimizers.schedules.PolynomialDecay(actor_lr, self.max_episode, 1e-10, power=1.0)
        self.critic_lr = tf.keras.optimizers.schedules.PolynomialDecay(critic_lr, self.max_episode, 1e-10, power=1.0)
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.actor_lr(self.episode))
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.critic_lr(self.episode))

    def show_logo(self):
        self.recorder.logger.info('''
　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　
　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘｘ　　ｘｘ　　　　　　　ｘｘｘ　　ｘｘ　　　　　
　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　ｘｘ　　　　ｘ　　　　　
　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　ｘｘ　　　　　　　　　　
　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　　　ｘ　　　ｘｘｘｘｘ　　　
　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘ　　　ｘｘｘ　　　　
　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘ　　　　ｘ　　　　　
　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘｘ　　ｘｘ　　　　　
　　　ｘｘｘｘｘｘｘxxx　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　　ｘｘｘｘｘｘ　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘ　　　　　
        ''')

    def choose_action(self, adj, x, visual_s, evaluation=False):
        a = self._get_action(adj, x, visual_s, evaluation).numpy()
        return a if self.is_continuous else sth.int2action_index(a, self.a_dim_or_list)

    @tf.function
    def _get_action(self, adj, x, visual_s, evaluation):
        adj,x , visual_s = self.cast(adj, x, visual_s)
        with tf.device(self.device):
            if self.is_continuous:
                mu = self.actor_net(adj, x, visual_s)
                pi = tf.clip_by_value(mu + self.action_noise(), -1, 1)
            else:
                logits = self.actor_net(adj, x, visual_s)
                mu = tf.argmax(logits, axis=1)
                cate_dist = tfp.distributions.Categorical(logits)
                pi = cate_dist.sample()
            if evaluation == True:
                return mu
            else:
                return pi

    def learn(self, **kwargs):
        self.episode = kwargs['episode']
        for i in range(kwargs['step']):
            if self.data.is_lg_batch_size:
                adj, x , visual_s, a, r, adj_, x_, visual_s_, done = self.data.sample()
                if self.use_priority:
                    self.IS_w = self.data.get_IS_w()
                td_error, summaries = self.train(adj, x , visual_s, a, r, adj_, x_, visual_s_, done)
                if self.use_priority:
                    td_error = np.squeeze(td_error.numpy())
                    self.data.update(td_error, self.episode)
                self.update_target_net_weights(
                    self.actor_target_net.weights + self.q_target_net.weights,
                    self.actor_net.weights + self.q_net.weights,
                    self.ployak)
                summaries.update(dict([
                    ['LEARNING_RATE/actor_lr', self.actor_lr(self.episode)],
                    ['LEARNING_RATE/critic_lr', self.critic_lr(self.episode)]
                ]))
                self.write_training_summaries(self.global_step, summaries)

    @tf.function(experimental_relax_shapes=True)
    def train(self, adj, x , visual_s, a, r, adj_, x_, visual_s_, done):#batch_size x num_agent x size
        adj, x , visual_s, a, r, adj_, x_, visual_s_, done = self.cast(adj, x , visual_s, a, r, adj_, x_, visual_s_, done)

        with tf.device(self.device):
            with tf.GradientTape() as tape:
                if self.is_continuous:
                    target_mu = self.actor_target_net(adj_, x_, visual_s_)
                    action_target = tf.clip_by_value(target_mu + self.action_noise(), -1, 1)
                else:
                    target_logits = self.actor_target_net(adj_, x_, visual_s_)
                    target_cate_dist = tfp.distributions.Categorical(target_logits)
                    target_pi = target_cate_dist.sample()
                    action_target = tf.one_hot(target_pi, self.a_counts, dtype=tf.float32)
                q = self.q_net(adj, x, visual_s, a)
                q_target = self.q_target_net(adj_, x_, visual_s_, action_target)
                dc_r = tf.stop_gradient(r + self.gamma * q_target * (1 - done))
                td_error = q - dc_r
                q_loss = 0.5 * tf.reduce_mean(tf.square(td_error) * self.IS_w)
            q_grads = tape.gradient(q_loss, self.q_net.tv)
            self.optimizer_critic.apply_gradients(
                zip(q_grads, self.q_net.tv)
            )
            with tf.GradientTape() as tape:

                if self.is_continuous:
                    mu = self.actor_net(adj, x, visual_s)
                    pi = tf.clip_by_value(mu + self.action_noise(), -1, 1)
                else:
                    logits = self.actor_net(adj, x , visual_s)
                    logp_all = tf.nn.log_softmax(logits)
                    gumbel_noise = tf.cast(self.gumbel_dist.sample([a.shape[0], self.a_counts]), dtype=tf.float32)
                    _pi = tf.nn.softmax((logp_all + gumbel_noise) / self.discrete_tau)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(_pi, axis=-1), self.a_counts)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    pi = _pi_diff + _pi
                q_actor = self.q_net(adj, x, visual_s, pi)
                actor_loss = -tf.reduce_mean(q_actor)

            actor_grads = tape.gradient(actor_loss, self.actor_net.tv)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_net.tv)
            )
            self.global_step.assign_add(1)
            return td_error, dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/critic_loss', q_loss],
                ['Statistics/q_min', tf.reduce_min(q)],
                ['Statistics/q_mean', tf.reduce_mean(q)],
                ['Statistics/q_max', tf.reduce_max(q)],
                ['Statistics/td_error', tf.reduce_mean(td_error)],
            ])

    @tf.function(experimental_relax_shapes=True)
    def train_persistent(self, adj, x , visual_s, a, r, adj_, x_, visual_s_, done):
        adj, x , visual_s, a, r, adj_, x_, visual_s_, done = self.cast(adj, x , visual_s, a, r, adj_, x_, visual_s_, done)
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                if self.is_continuous:
                    target_mu = self.actor_target_net(adj_, x_, visual_s_)
                    action_target = tf.clip_by_value(target_mu + self.action_noise(), -1, 1)
                    mu = self.actor_net(adj, x, visual_s)
                    pi = tf.clip_by_value(mu + self.action_noise(), -1, 1)
                else:
                    target_logits = self.actor_target_net(adj_, x_, visual_s_)
                    target_cate_dist = tfp.distributions.Categorical(target_logits)
                    target_pi = target_cate_dist.sample()
                    action_target = tf.one_hot(target_pi, self.a_counts, dtype=tf.float32)
                    logits = self.actor_net(adj, x, visual_s)
                    logp_all = tf.nn.log_softmax(logits)
                    gumbel_noise = tf.cast(self.gumbel_dist.sample([a.shape[0], self.a_counts]), dtype=tf.float32)
                    _pi = tf.nn.softmax((logp_all + gumbel_noise) / self.discrete_tau)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(_pi, axis=-1), self.a_counts)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    pi = _pi_diff + _pi
                q = self.q_net(adj, x, visual_s, a)
                q_target = self.q_target_net(adj_, x_, visual_s_, action_target)
                dc_r = tf.stop_gradient(r + self.gamma * q_target * (1 - done))
                td_error = q - dc_r
                q_loss = 0.5 * tf.reduce_mean(tf.square(td_error) * self.IS_w)

                q_actor = self.q_net(adj, x, visual_s, pi)
                actor_loss = -tf.reduce_mean(q_actor)
            q_grads = tape.gradient(q_loss, self.q_net.tv)
            self.optimizer_critic.apply_gradients(
                zip(q_grads, self.q_net.tv)
            )
            actor_grads = tape.gradient(actor_loss, self.actor_net.tv)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_net.tv)
            )
            self.global_step.assign_add(1)
            return td_error, dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/critic_loss', q_loss],
                ['Statistics/q_min', tf.reduce_min(q)],
                ['Statistics/q_mean', tf.reduce_mean(q)],
                ['Statistics/q_max', tf.reduce_max(q)]
            ])
