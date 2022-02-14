"""
Architecture for Few-Shot Regression (Sinusoid and Multimodal)
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import flags
FLAGS = flags.FLAGS
# from .BaseModel import BaseModel


#Define Feature Extractor
class FeatureExtractor(object):
    
    #Initialize feature extractor depth and width based on the task
    def __init__(self, inputs):
        if FLAGS.datasource == 'spindle':
            ## The number of nodes, (64 in this case) needs to match self.hidden below in FFN
            self.n_units = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
        with tf.variable_scope("extractor", reuse=tf.AUTO_REUSE):
            self.build_model(inputs)

    #Build model layers according to the defined size
    def build_model(self, inputs):
        running_output = inputs
        for i, size in enumerate(self.n_units[:-1]):
            running_output = tf.nn.relu(
                tf.layers.dense(running_output, size, name="layer_{}".format(i)))
            # Last layer without a ReLu
        running_output = tf.layers.dense(
            running_output, self.n_units[-1], name="layer_{}".format(i + 1))
        
        #Feature extractor output
        self.output = running_output  # shape = (meta_batch_size, num_shot_train, number nodes in final layer)

#Define Weights generator
class WeightsGenerator(object):
    
    #Initialize Weights generator
    def __init__(self, inputs, hidden, attention_layers):
        
        #Define size and shape of weights generator based on the task
        if FLAGS.datasource == 'spindle':
            ## this value needs to 2x larger than self.hidden below in FFN
             output_units = 128
        with tf.variable_scope("attention"):
            
            train_embed = inputs
            for i in np.arange(attention_layers):
                query = tf.layers.dense(inputs=train_embed, units=hidden, activation=None, name="query_{}".format(i))
                key = tf.layers.dense(inputs=train_embed, units=hidden, activation=None, name="key_{}".format(i))
                value = tf.layers.dense(inputs=train_embed, units=hidden, activation=None, name="value_{}".format(i))
                train_embed, _ = self.attention(query, key, value)
                dense = tf.layers.dense(inputs=train_embed, units=hidden*2, activation=tf.nn.relu, name="ff_layer{}_dense0".format(i))
                train_embed += tf.layers.dense(inputs=dense, units=hidden, activation=None, name="ff_layer{}_dense1".format(i))
                train_embed = tf.contrib.layers.layer_norm(train_embed, begin_norm_axis=2)

            train_embed = tf.layers.dense(
                inputs=train_embed,
                units=output_units,
                activation=None,
            )
            self.final_weights = tf.reduce_mean(train_embed, axis=1, keep_dims=True)

    #define attention function to be used during weights generator evaluation
    def attention(self, query, key, value):
        dotp = tf.matmul(query, key, transpose_b=True) / (tf.cast(tf.shape(query)[-1], tf.float32) ** 0.5)
        attention_weights = tf.nn.softmax(dotp)
        weighted_sum = tf.matmul(attention_weights, value)
        output = weighted_sum + query
        output = tf.contrib.layers.layer_norm(output, begin_norm_axis=2)
        return output, attention_weights

    def mlp(self, input, output_sizes, name):
        """Apply MLP to the final axis of a 3D tensor (reusing already defined MLPs).

        Args:
          input: input tensor of shape [B,n,d_in].
          output_sizes: An iterable containing the output sizes of the MLP as defined
              in `basic.Linear`.
          variable_scope: String giving the name of the variable scope. If this is set
              to be the same as a previously defined MLP, then the weights are reused.

        Returns:
          tensor of shape [B,n,d_out] where d_out=output_sizes[-1]
        """
        output = input
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for i, size in enumerate(output_sizes[:-1]):
                output = tf.nn.relu(
                    tf.layers.dense(output, size, name="layer_{}".format(i), use_bias=False))
            # Last layer without a ReLu
            output = tf.layers.dense(
                output, output_sizes[-1], name="layer_{}".format(i + 1), use_bias=False)
        return output


#Create the model    
class FFN():
    
    #Initialize model based on the given inputs
    def __init__(self, name, num_train_samples, num_test_samples, l1_penalty, l2_penalty):
        # super(FFN, self).__init__()
        self.name = name
        # Attention parameters
        self.attention_layers = 4
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.hidden = 64
        
        #Define variables for training
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.build_model(num_train_samples, num_test_samples)
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            self.saver = tf.train.Saver(var_list=variables, max_to_keep=3)
    
    #Build model based on the task and data
    def build_model(self, num_train_samples, num_test_samples):
       
        self.train_inputs = tf.placeholder(
            shape=(None, None, 1 if FLAGS.datasource == 'sinusoid' else 3),
            dtype=tf.float32,
            name="train_inputs",
        )
        self.train_labels = tf.placeholder(
            shape=(None, None, 2),
            dtype=tf.float32,
            name="train_labels",
        )
        self.test_inputs = tf.placeholder(
            shape=(None, None, 1 if FLAGS.datasource == 'sinusoid' else 3),
            dtype=tf.float32,
            name="test_inputs"
        )
        self.test_labels = tf.placeholder(
            shape=(None, None, 2),
            dtype=tf.float32,
            name="test_labels",
        )

        
        # Extract training features
        self.train_feature_extractor = FeatureExtractor(self.train_inputs)
        
        #Concatenate data label to feature vector before passing it into the weights generator
        t_input = tf.concat([self.train_feature_extractor.output, self.train_labels], axis=-1)
        
        #Run weights generator function
        weights_generator_train = WeightsGenerator(t_input, self.hidden, self.attention_layers)
        
        
        #Extract the mean weights vector to be used for evaluation
        self.train_final_weights = weights_generator_train.final_weights
        
        # Extract test features
        test_feature_extractor = FeatureExtractor(self.test_inputs)
        
        #Set the test features
        self.test_features = test_feature_extractor.output
        
        #Prediction list
        self.p1_list = []
        self.p2_list = []
        
        #Label list
        self.label_list1 = []
        self.label_list2 = []
        
        #Dot product evaluation between test features and mean weights
        
        self.predictions1 = tf.matmul(self.test_features, self.train_final_weights[:,:,0:64], transpose_b=True)
        self.predictions2 = tf.matmul(self.test_features, self.train_final_weights[:,:,64:128], transpose_b=True)
        
        self.p1_list.append(self.predictions1)
        self.p2_list.append(self.predictions2)
        
        self.label_list1.append(self.test_labels[:,:,0])
        self.label_list2.append(self.test_labels[:,:,1])
        
        #Calculate Losses
        self.penalty_loss_amplitude = self.l1_penalty * tf.norm(self.train_final_weights[:,:,0:64], ord=1) + \
                            self.l2_penalty * tf.norm(self.train_final_weights[:,:,0:64], ord = 2)
        
        self.penalty_loss_phase = self.l1_penalty * tf.norm(self.train_final_weights[:,:,64:128], ord = 1) + \
                            self.l2_penalty * tf.norm(self.train_final_weights[:,:,64:128], ord = 2)
        
       
        self.loss_amplitude = tf.losses.mean_squared_error(labels=tf.reshape(self.test_labels[:,:,0], [-1]), \
                                                           predictions=tf.reshape(self.predictions1, [-1])) + self.penalty_loss_amplitude
        
        self.loss_phase = tf.losses.mean_squared_error(labels=tf.reshape(self.test_labels[:,:,1], [-1]), \
                                                       predictions=tf.reshape(self.predictions2, [-1])) + self.penalty_loss_phase
        
        self.loss = self.loss_amplitude + self.loss_phase
        
        self.plain_loss_amplitude = tf.losses.mean_squared_error(labels=tf.reshape(self.test_labels[:,:,0], [-1]),\
                                                                 predictions=tf.reshape(self.predictions1, [-1]))  
        
        self.plain_loss_phase = tf.losses.mean_squared_error(labels=tf.reshape(self.test_labels[:,:,1], [-1]), \
                                                             predictions=tf.reshape(self.predictions2, [-1]))




