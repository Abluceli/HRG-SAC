import logging
from enum import Enum
from typing import Any, Callable, Dict, List

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as c_layers

from mlagents.trainers.trainer import UnityTrainerException

logger = logging.getLogger("mlagents.trainers")

ActivationFunction = Callable[[tf.Tensor], tf.Tensor]

EPSILON = 1e-7


class EncoderType(Enum):
    SIMPLE = "simple"
    NATURE_CNN = "nature_cnn"
    RESNET = "resnet"


class LearningRateSchedule(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"


class LearningModel(object):
    _version_number_ = 2

    def __init__(
        self, m_size, normalize, use_recurrent, brain, seed, stream_names=None
    ):
        tf.set_random_seed(seed)
        self.brain = brain
        self.vector_in = None
        self.global_step, self.increment_step, self.steps_to_increment = (
            self.create_global_steps()
        )
        self.visual_in = []
        self.batch_size = tf.placeholder(shape=None, dtype=tf.int32, name="batch_size")
        self.sequence_length = tf.placeholder(
            shape=None, dtype=tf.int32, name="sequence_length"
        )
        self.mask_input = tf.placeholder(shape=[None], dtype=tf.float32, name="masks")
        self.mask = tf.cast(self.mask_input, tf.int32)
        self.stream_names = stream_names or []
        self.use_recurrent = use_recurrent
        if self.use_recurrent:
            self.m_size = m_size
        else:
            self.m_size = 0
        self.normalize = normalize
        self.act_size = brain.vector_action_space_size
        self.vec_obs_size = (
            brain.vector_observation_space_size * brain.num_stacked_vector_observations
        )
        self.vis_obs_size = brain.number_visual_observations
        tf.Variable(
            int(brain.vector_action_space_type == "continuous"),
            name="is_continuous_control",
            trainable=False,
            dtype=tf.int32,
        )
        tf.Variable(
            self._version_number_,
            name="version_number",
            trainable=False,
            dtype=tf.int32,
        )
        tf.Variable(self.m_size, name="memory_size", trainable=False, dtype=tf.int32)
        if brain.vector_action_space_type == "continuous":
            tf.Variable(
                self.act_size[0],
                name="action_output_shape",
                trainable=False,
                dtype=tf.int32,
            )
        else:
            tf.Variable(
                sum(self.act_size),
                name="action_output_shape",
                trainable=False,
                dtype=tf.int32,
            )

    @staticmethod
    def create_global_steps():
        """Creates TF ops to track and increment global training step."""
        global_step = tf.Variable(
            0, name="global_step", trainable=False, dtype=tf.int32
        )
        steps_to_increment = tf.placeholder(
            shape=[], dtype=tf.int32, name="steps_to_increment"
        )
        increment_step = tf.assign(global_step, tf.add(global_step, steps_to_increment))
        return global_step, increment_step, steps_to_increment

    @staticmethod
    def create_learning_rate(
        lr_schedule: LearningRateSchedule,
        lr: float,
        global_step: tf.Tensor,
        max_step: int,
    ) -> tf.Tensor:
        if lr_schedule == LearningRateSchedule.CONSTANT:
            learning_rate = tf.Variable(lr)
        elif lr_schedule == LearningRateSchedule.LINEAR:
            learning_rate = tf.train.polynomial_decay(
                lr, global_step, max_step, 1e-10, power=1.0
            )
        else:
            raise UnityTrainerException(
                "The learning rate schedule {} is invalid.".format(lr_schedule)
            )
        return learning_rate

    @staticmethod
    def scaled_init(scale):
        return c_layers.variance_scaling_initializer(scale)

    @staticmethod
    def swish(input_activation: tf.Tensor) -> tf.Tensor:
        """Swish activation function. For more info: https://arxiv.org/abs/1710.05941"""
        return tf.multiply(input_activation, tf.nn.sigmoid(input_activation))

    @staticmethod
    def create_visual_input(camera_parameters: Dict[str, Any], name: str) -> tf.Tensor:
        """
        Creates image input op.
        :param camera_parameters: Parameters for visual observation from BrainInfo.
        :param name: Desired name of input op.
        :return: input op.
        """
        o_size_h = camera_parameters["height"]
        o_size_w = camera_parameters["width"]
        bw = camera_parameters["blackAndWhite"]

        if bw:
            c_channels = 1
        else:
            c_channels = 3

        visual_in = tf.placeholder(
            shape=[None, o_size_h, o_size_w, c_channels], dtype=tf.float32, name=name
        )
        return visual_in

    def create_vector_input(self, name="vector_observation"):
        """
        Creates ops for vector observation input.
        :param name: Name of the placeholder op.
        :param vec_obs_size: Size of stacked vector observation.
        :return:
        """
        self.vector_in = tf.placeholder(
            shape=[None, self.vec_obs_size], dtype=tf.float32, name=name
        )
        if self.normalize:
            self.create_normalizer(self.vector_in)
            return self.normalize_vector_obs(self.vector_in)
        else:
            return self.vector_in

    def normalize_vector_obs(self, vector_obs):
        normalized_state = tf.clip_by_value(
            (vector_obs - self.running_mean)
            / tf.sqrt(
                self.running_variance
                / (tf.cast(self.normalization_steps, tf.float32) + 1)
            ),
            -5,
            5,
            name="normalized_state",
        )
        return normalized_state

    def create_normalizer(self, vector_obs):
        self.normalization_steps = tf.get_variable(
            "normalization_steps",
            [],
            trainable=False,
            dtype=tf.int32,
            initializer=tf.ones_initializer(),
        )
        self.running_mean = tf.get_variable(
            "running_mean",
            [self.vec_obs_size],
            trainable=False,
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
        )
        self.running_variance = tf.get_variable(
            "running_variance",
            [self.vec_obs_size],
            trainable=False,
            dtype=tf.float32,
            initializer=tf.ones_initializer(),
        )
        self.update_normalization = self.create_normalizer_update(vector_obs)

    def create_normalizer_update(self, vector_input):
        mean_current_observation = tf.reduce_mean(vector_input, axis=0)
        new_mean = self.running_mean + (
            mean_current_observation - self.running_mean
        ) / tf.cast(tf.add(self.normalization_steps, 1), tf.float32)
        new_variance = self.running_variance + (mean_current_observation - new_mean) * (
            mean_current_observation - self.running_mean
        )
        update_mean = tf.assign(self.running_mean, new_mean)
        update_variance = tf.assign(self.running_variance, new_variance)
        update_norm_step = tf.assign(
            self.normalization_steps, self.normalization_steps + 1
        )
        return tf.group([update_mean, update_variance, update_norm_step])

    @staticmethod
    def create_vector_observation_encoder(
        observation_input: tf.Tensor,
        h_size: int,
        activation: ActivationFunction,
        num_layers: int,
        scope: str,
        reuse: bool,
    ) -> tf.Tensor:
        """
        Builds a set of hidden state encoders.
        :param reuse: Whether to re-use the weights within the same scope.
        :param scope: Graph scope for the encoder ops.
        :param observation_input: Input vector.
        :param h_size: Hidden layer size.
        :param activation: What type of activation function to use for layers.
        :param num_layers: number of hidden layers to create.
        :return: List of hidden layer tensors.
        """
        with tf.variable_scope(scope):
            hidden = observation_input
            for i in range(num_layers):
                hidden = tf.layers.dense(
                    hidden,
                    h_size,
                    activation=activation,
                    reuse=reuse,
                    name="hidden_{}".format(i),
                    kernel_initializer=c_layers.variance_scaling_initializer(1.0),
                )
        return hidden

    def create_visual_observation_encoder(
        self,
        image_input: tf.Tensor,
        h_size: int,
        activation: ActivationFunction,
        num_layers: int,
        scope: str,
        reuse: bool,
    ) -> tf.Tensor:
        """
        Builds a set of resnet visual encoders.
        :param image_input: The placeholder for the image input to use.
        :param h_size: Hidden layer size.
        :param activation: What type of activation function to use for layers.
        :param num_layers: number of hidden layers to create.
        :param scope: The scope of the graph within which to create the ops.
        :param reuse: Whether to re-use the weights within the same scope.
        :return: List of hidden layer tensors.
        """
        with tf.variable_scope(scope):
            conv1 = tf.layers.conv2d(
                image_input,
                16,
                kernel_size=[8, 8],
                strides=[4, 4],
                activation=tf.nn.elu,
                reuse=reuse,
                name="conv_1",
            )
            conv2 = tf.layers.conv2d(
                conv1,
                32,
                kernel_size=[4, 4],
                strides=[2, 2],
                activation=tf.nn.elu,
                reuse=reuse,
                name="conv_2",
            )
            hidden = c_layers.flatten(conv2)

        with tf.variable_scope(scope + "/" + "flat_encoding"):
            hidden_flat = self.create_vector_observation_encoder(
                hidden, h_size, activation, num_layers, scope, reuse
            )
        return hidden_flat

    def create_nature_cnn_visual_observation_encoder(
        self,
        image_input: tf.Tensor,
        h_size: int,
        activation: ActivationFunction,
        num_layers: int,
        scope: str,
        reuse: bool,
    ) -> tf.Tensor:
        """
        Builds a set of resnet visual encoders.
        :param image_input: The placeholder for the image input to use.
        :param h_size: Hidden layer size.
        :param activation: What type of activation function to use for layers.
        :param num_layers: number of hidden layers to create.
        :param scope: The scope of the graph within which to create the ops.
        :param reuse: Whether to re-use the weights within the same scope.
        :return: List of hidden layer tensors.
        """
        with tf.variable_scope(scope):
            conv1 = tf.layers.conv2d(
                image_input,
                32,
                kernel_size=[8, 8],
                strides=[4, 4],
                activation=tf.nn.elu,
                reuse=reuse,
                name="conv_1",
            )
            conv2 = tf.layers.conv2d(
                conv1,
                64,
                kernel_size=[4, 4],
                strides=[2, 2],
                activation=tf.nn.elu,
                reuse=reuse,
                name="conv_2",
            )
            conv3 = tf.layers.conv2d(
                conv2,
                64,
                kernel_size=[3, 3],
                strides=[1, 1],
                activation=tf.nn.elu,
                reuse=reuse,
                name="conv_3",
            )
            hidden = c_layers.flatten(conv3)

        with tf.variable_scope(scope + "/" + "flat_encoding"):
            hidden_flat = self.create_vector_observation_encoder(
                hidden, h_size, activation, num_layers, scope, reuse
            )
        return hidden_flat

    def create_resnet_visual_observation_encoder(
        self,
        image_input: tf.Tensor,
        h_size: int,
        activation: ActivationFunction,
        num_layers: int,
        scope: str,
        reuse: bool,
    ) -> tf.Tensor:
        """
        Builds a set of resnet visual encoders.
        :param image_input: The placeholder for the image input to use.
        :param h_size: Hidden layer size.
        :param activation: What type of activation function to use for layers.
        :param num_layers: number of hidden layers to create.
        :param scope: The scope of the graph within which to create the ops.
        :param reuse: Whether to re-use the weights within the same scope.
        :return: List of hidden layer tensors.
        """
        n_channels = [16, 32, 32]  # channel for each stack
        n_blocks = 2  # number of residual blocks
        with tf.variable_scope(scope):
            hidden = image_input
            for i, ch in enumerate(n_channels):
                hidden = tf.layers.conv2d(
                    hidden,
                    ch,
                    kernel_size=[3, 3],
                    strides=[1, 1],
                    reuse=reuse,
                    name="layer%dconv_1" % i,
                )
                hidden = tf.layers.max_pooling2d(
                    hidden, pool_size=[3, 3], strides=[2, 2], padding="same"
                )
                # create residual blocks
                for j in range(n_blocks):
                    block_input = hidden
                    hidden = tf.nn.relu(hidden)
                    hidden = tf.layers.conv2d(
                        hidden,
                        ch,
                        kernel_size=[3, 3],
                        strides=[1, 1],
                        padding="same",
                        reuse=reuse,
                        name="layer%d_%d_conv1" % (i, j),
                    )
                    hidden = tf.nn.relu(hidden)
                    hidden = tf.layers.conv2d(
                        hidden,
                        ch,
                        kernel_size=[3, 3],
                        strides=[1, 1],
                        padding="same",
                        reuse=reuse,
                        name="layer%d_%d_conv2" % (i, j),
                    )
                    hidden = tf.add(block_input, hidden)
            hidden = tf.nn.relu(hidden)
            hidden = c_layers.flatten(hidden)

        with tf.variable_scope(scope + "/" + "flat_encoding"):
            hidden_flat = self.create_vector_observation_encoder(
                hidden, h_size, activation, num_layers, scope, reuse
            )
        return hidden_flat

    @staticmethod
    def create_discrete_action_masking_layer(all_logits, action_masks, action_size):
        """
        Creates a masking layer for the discrete actions
        :param all_logits: The concatenated unnormalized action probabilities for all branches
        :param action_masks: The mask for the logits. Must be of dimension [None x total_number_of_action]
        :param action_size: A list containing the number of possible actions for each branch
        :return: The action output dimension [batch_size, num_branches], the concatenated
            normalized probs (after softmax)
        and the concatenated normalized log probs
        """
        action_idx = [0] + list(np.cumsum(action_size))
        branches_logits = [
            all_logits[:, action_idx[i] : action_idx[i + 1]]
            for i in range(len(action_size))
        ]
        branch_masks = [
            action_masks[:, action_idx[i] : action_idx[i + 1]]
            for i in range(len(action_size))
        ]
        raw_probs = [
            tf.multiply(tf.nn.softmax(branches_logits[k]) + EPSILON, branch_masks[k])
            for k in range(len(action_size))
        ]
        normalized_probs = [
            tf.divide(raw_probs[k], tf.reduce_sum(raw_probs[k], axis=1, keepdims=True))
            for k in range(len(action_size))
        ]
        output = tf.concat(
            [
                tf.multinomial(tf.log(normalized_probs[k] + EPSILON), 1)
                for k in range(len(action_size))
            ],
            axis=1,
        )
        return (
            output,
            tf.concat([normalized_probs[k] for k in range(len(action_size))], axis=1),
            tf.concat(
                [
                    tf.log(normalized_probs[k] + EPSILON)
                    for k in range(len(action_size))
                ],
                axis=1,
            ),
        )

    def create_observation_streams(
        self,
        num_streams: int,
        h_size: int,
        num_layers: int,
        vis_encode_type: EncoderType = EncoderType.SIMPLE,
        stream_scopes: List[str] = None,
    ) -> tf.Tensor:
        """
        Creates encoding stream for observations.
        :param num_streams: Number of streams to create.
        :param h_size: Size of hidden linear layers in stream.
        :param num_layers: Number of hidden linear layers in stream.
        :param stream_scopes: List of strings (length == num_streams), which contains
            the scopes for each of the streams. None if all under the same TF scope.
        :return: List of encoded streams.
        """
        brain = self.brain
        activation_fn = self.swish

        self.visual_in = []
        for i in range(brain.number_visual_observations):
            visual_input = self.create_visual_input(
                brain.camera_resolutions[i], name="visual_observation_" + str(i)
            )
            self.visual_in.append(visual_input)
        vector_observation_input = self.create_vector_input()

        final_hiddens = []
        for i in range(num_streams):
            visual_encoders = []
            hidden_state, hidden_visual = None, None
            _scope_add = stream_scopes[i] if stream_scopes else ""
            if self.vis_obs_size > 0:
                if vis_encode_type == EncoderType.RESNET:
                    for j in range(brain.number_visual_observations):
                        encoded_visual = self.create_resnet_visual_observation_encoder(
                            self.visual_in[j],
                            h_size,
                            activation_fn,
                            num_layers,
                            _scope_add + "main_graph_{}_encoder{}".format(i, j),
                            False,
                        )
                        visual_encoders.append(encoded_visual)
                elif vis_encode_type == EncoderType.NATURE_CNN:
                    for j in range(brain.number_visual_observations):
                        encoded_visual = self.create_nature_cnn_visual_observation_encoder(
                            self.visual_in[j],
                            h_size,
                            activation_fn,
                            num_layers,
                            _scope_add + "main_graph_{}_encoder{}".format(i, j),
                            False,
                        )
                        visual_encoders.append(encoded_visual)
                else:
                    for j in range(brain.number_visual_observations):
                        encoded_visual = self.create_visual_observation_encoder(
                            self.visual_in[j],
                            h_size,
                            activation_fn,
                            num_layers,
                            _scope_add + "main_graph_{}_encoder{}".format(i, j),
                            False,
                        )
                        visual_encoders.append(encoded_visual)
                hidden_visual = tf.concat(visual_encoders, axis=1)
            if brain.vector_observation_space_size > 0:
                hidden_state = self.create_vector_observation_encoder(
                    vector_observation_input,
                    h_size,
                    activation_fn,
                    num_layers,
                    _scope_add + "main_graph_{}".format(i),
                    False,
                )
            if hidden_state is not None and hidden_visual is not None:
                final_hidden = tf.concat([hidden_visual, hidden_state], axis=1)
            elif hidden_state is None and hidden_visual is not None:
                final_hidden = hidden_visual
            elif hidden_state is not None and hidden_visual is None:
                final_hidden = hidden_state
            else:
                raise Exception(
                    "No valid network configuration possible. "
                    "There are no states or observations in this brain"
                )
            final_hiddens.append(final_hidden)
        return final_hiddens

    @staticmethod
    def create_recurrent_encoder(input_state, memory_in, sequence_length, name="lstm"):
        """
        Builds a recurrent encoder for either state or observations (LSTM).
        :param sequence_length: Length of sequence to unroll.
        :param input_state: The input tensor to the LSTM cell.
        :param memory_in: The input memory to the LSTM cell.
        :param name: The scope of the LSTM cell.
        """
        s_size = input_state.get_shape().as_list()[1]
        m_size = memory_in.get_shape().as_list()[1]
        lstm_input_state = tf.reshape(input_state, shape=[-1, sequence_length, s_size])
        memory_in = tf.reshape(memory_in[:, :], [-1, m_size])
        half_point = int(m_size / 2)
        with tf.variable_scope(name):
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(half_point)
            lstm_vector_in = tf.contrib.rnn.LSTMStateTuple(
                memory_in[:, :half_point], memory_in[:, half_point:]
            )
            recurrent_output, lstm_state_out = tf.nn.dynamic_rnn(
                rnn_cell, lstm_input_state, initial_state=lstm_vector_in
            )

        recurrent_output = tf.reshape(recurrent_output, shape=[-1, half_point])
        return recurrent_output, tf.concat([lstm_state_out.c, lstm_state_out.h], axis=1)

    def create_value_heads(self, stream_names, hidden_input):
        """
        Creates one value estimator head for each reward signal in stream_names.
        Also creates the node corresponding to the mean of all the value heads in self.value.
        self.value_head is a dictionary of stream name to node containing the value estimator head for that signal.
        :param stream_names: The list of reward signal names
        :param hidden_input: The last layer of the Critic. The heads will consist of one dense hidden layer on top
        of the hidden input.
        """
        self.value_heads = {}
        for name in stream_names:
            value = tf.layers.dense(hidden_input, 1, name="{}_value".format(name))
            self.value_heads[name] = value
        self.value = tf.reduce_mean(list(self.value_heads.values()), 0)
