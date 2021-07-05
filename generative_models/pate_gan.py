"""
A generative model training algorithm based on
"PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees"
by J. Yoon, J. Jordon, M. van der Schaar, published in International Conference on Learning Representations (ICLR), 2019
Adapted from: https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/82d7f91d46db54d256ff4fc920d513499ddd2ab8/alg/pategan/
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from pandas import DataFrame

from generative_models.generative_model import GenerativeModel
from utils.logging import LOGGER
from utils.constants import *


ZERO_TOL = 1e-8


class PATEGAN(GenerativeModel):
    """ A generative adversarial network trained under the PATE framework to achieve differential privacy """

    def __init__(self, metadata,
                 eps=1, delta=1e-5, infer_ranges=False,
                 num_teachers=10, n_iters=100, batch_size=128,
                 learning_rate=1e-4, multiprocess=False):
        """
        :param metadata: dict: Attribute metadata describing the data domain of the synthetic target data
        :param eps: float: Privacy parameter
        :param delta: float: Privacy parameter
        :param target: str: Name of the target variable for downstream classification tasks
        :param num_teachers: int: Number of teacher discriminators
        :param n_iters: int: Number of training iterations
        """
        # Data description
        self.metadata, self.attribute_list = self.read_meta(metadata)
        self.datatype = DataFrame
        self.nfeatures = self.get_num_features()

        # Privacy params
        self.epsilon = eps
        self.delta = delta
        self.infer_ranges = infer_ranges

        # Training params
        self.num_teachers = num_teachers
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.z_dim = int(self.nfeatures / 4)
        self.h_dim = int(self.nfeatures)

        # Configure device
        device_name = tf.test.gpu_device_name()
        if device_name is '':
            self.device_spec = tf.DeviceSpec(device_type='CPU', device_index=0)
        else:
            self.device_spec = tf.DeviceSpec(device_type='GPU', device_index=0)

        with tf.device(self.device_spec.to_string()):
            # Variable init
            # Feature matrix
            self.X = tf.placeholder(tf.float32, shape=[None, self.nfeatures])
            # Latent space
            self.Z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
            # Noise variable
            self.M = tf.placeholder(tf.float32, shape=[None, 1])
            # Generator
            self.GDist = None
            self._generator()
            # Discriminator
            self._discriminator()
            self.sess = tf.Session()

        self.multiprocess = multiprocess

        self.trained = False

        self.__name__ = f'PateGanEps{self.epsilon}'

    @property
    def laplace_noise_scale(self):
        return np.sqrt(2 * np.log(1.25 * 10**self.delta)) / self.epsilon

    def get_num_features(self):
        nfeatures = 0

        for cname, cdict in self.metadata.items():
            data_type = cdict['type']
            if data_type == FLOAT or data_type == INTEGER:
                nfeatures += 1

            elif data_type == CATEGORICAL or data_type == ORDINAL:
                nfeatures += len(cdict['categories'])

            else:
                raise ValueError(f'Unkown data type {data_type} for attribute {cname}')

        return nfeatures

    def read_meta(self, metadata):
        meta_dict = {}
        attr_names = []
        for cdict in metadata['columns']:
            attr_name = cdict['name']
            data_type = cdict['type']
            if data_type == FLOAT or data_type == INTEGER:
                meta_dict[attr_name] = {
                    'type': data_type,
                    'min': cdict['min'],
                    'max': cdict['max']
                }

            elif data_type == CATEGORICAL or data_type == ORDINAL:
                meta_dict[attr_name] = {
                    'type': data_type,
                    'categories': cdict['i2s']
                }

            else:
                raise ValueError(f'Unknown data type {data_type} for attribute {attr_name}')

            attr_names.append(attr_name)

        return meta_dict, attr_names

    def _generator(self):
        self.G_W1 = tf.Variable(self._xavier_init([self.z_dim, self.h_dim]))
        self.G_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]))

        self.G_W2 = tf.Variable(self._xavier_init([self.h_dim, self.h_dim]))
        self.G_b2 = tf.Variable(tf.zeros(shape=[self.h_dim]))

        self.G_W3 = tf.Variable(self._xavier_init([self.h_dim, self.nfeatures]))
        self.G_b3 = tf.Variable(tf.zeros(shape=[self.nfeatures]))

        self.theta_G = [self.G_W1, self.G_W2, self.G_W3, self.G_b1, self.G_b2, self.G_b3]

    def _discriminator(self):
        self.D_W1 = tf.Variable(self._xavier_init([self.nfeatures, self.h_dim]))
        self.D_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]))

        self.D_W2 = tf.Variable(self._xavier_init([self.h_dim, self.h_dim]))
        self.D_b2 = tf.Variable(tf.zeros(shape=[self.h_dim]))

        self.D_W3 = tf.Variable(self._xavier_init([self.h_dim, 1]))
        self.D_b3 = tf.Variable(tf.zeros(shape=[1]))

        self.theta_D = [self.D_W1, self.D_W2, self.D_W3, self.D_b1, self.D_b2, self.D_b3]

    def fit(self, data):
        """Fit a generative model of the training data distribution.
        :param data: DataFrame: Training set
        """
        assert isinstance(data, self.datatype), f'{self.__class__.__name__} expects {self.datatype} as input data but got {type(data)}'

        # Clean up
        if self.trained:
            self._generator()
            self._discriminator()
            self.sess = tf.Session()
            self.trained = False

        LOGGER.debug(f'Start fitting {self.__class__.__name__} to data of shape {data.shape}...')
        nsamples = len(data)
        features_train = self._encode_data(data)

        with tf.device(self.device_spec.to_string()):
            # Generator
            self.GDist = self.gen_out(self.Z)

            # Discriminator
            D_real = self.discriminator_out(self.X)
            D_fake = self.discriminator_out(self.GDist)
            D_entire = tf.concat(axis=0, values=[D_real, D_fake])

            # Replacement of Clipping algorithm to Penalty term
            # 1. Line 6 in Algorithm 1
            noisy_vals = tf.random_uniform([self.batch_size, 1], minval=0., maxval=1.)
            X_inter = noisy_vals * self.X + (1. - noisy_vals) * self.GDist

            # 2. Line 7 in Algorithm 1
            grad = tf.gradients(self.discriminator_out(X_inter), [X_inter])[0]
            grad_norm = tf.sqrt(tf.reduce_sum(grad ** 2 + ZERO_TOL, axis=1))
            grad_pen = self.num_teachers * tf.reduce_mean((grad_norm - 1) ** 2)

            # Loss function
            discriminator_loss = tf.reduce_mean((1 - self.M) * D_entire) - tf.reduce_mean(self.M * D_entire) + grad_pen
            generator_loss = -tf.reduce_mean(D_fake)

            # Solver
            discriminator_solver = (tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(discriminator_loss, var_list=self.theta_D))
            generator_solver = (tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(generator_loss, var_list=self.theta_G))

            # Start session
            self.sess.run(tf.global_variables_initializer())

            # Training iterations
            for _ in range(self.n_iters):
                # TODO: Move dataset splitting here
                # For fixed generator weights run teacher training
                for _ in range(self.num_teachers):
                    # Sample latent vars
                    latent_batch = self._sample_latent_z(self.batch_size, self.z_dim)

                    # Sample real
                    train_idx_teach = self._sample_real_x(nsamples, self.batch_size) # Does this way of sampling satisfy DP? Should be disjoint subsets!
                    features_train_batch = features_train[train_idx_teach, :]

                    labels_real = np.ones([self.batch_size, ])
                    labels_fake = np.zeros([self.batch_size, ])

                    labels_batch = np.concatenate((labels_real, labels_fake), 0)

                    gaussian_noise = np.random.normal(loc=0.0, scale=self.laplace_noise_scale, size=self.batch_size * 2)

                    labels_batch = labels_batch + gaussian_noise

                    labels_batch = (labels_batch > 0.5)

                    labels_batch = np.reshape(labels_batch.astype(float), (2 * self.batch_size, 1))

                    _, discriminator_loss_iter = self.sess.run([discriminator_solver, discriminator_loss], feed_dict={self.X: features_train_batch, self.Z: latent_batch, self.M: labels_batch})

                # Update generator weights
                latent_batch = self._sample_latent_z(self.batch_size, self.z_dim)

                _, generator_loss_iter = self.sess.run([generator_solver, generator_loss], feed_dict={self.Z: latent_batch})

        self.trained = True

    def generate_samples(self, nsamples):
        """""
        Samples synthetic data records from the fitted generative distribution
        :param nsamples: int: Number of synthetic records to generate
        :return synData: DataFrame: A synthetic dataset
        """
        with tf.device(self.device_spec.to_string()):
            # Output generation
            features_synthetic_encoded = self.sess.run([self.GDist], feed_dict={self.Z: self._sample_latent_z(nsamples, self.z_dim)})[0]

        # Revers numerical encoding
        synthetic_data = self._decode_data(features_synthetic_encoded)
        synthetic_data = synthetic_data.iloc[np.random.permutation(synthetic_data.index)].reset_index(drop=True)

        return synthetic_data


    def gen_out(self, z):
        G_h1 = tf.nn.tanh(tf.matmul(z, self.G_W1) + self.G_b1)
        G_h2 = tf.nn.tanh(tf.matmul(G_h1, self.G_W2) + self.G_b2)
        G_log_prob = tf.nn.sigmoid(tf.matmul(G_h2, self.G_W3) + self.G_b3)

        return G_log_prob

    def discriminator_out(self, x):
        D_h1 = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, self.D_W2) + self.D_b2)
        out = (tf.matmul(D_h2, self.D_W3) + self.D_b3)

        return out

    def _xavier_init(self,size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)

        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def _sample_latent_z(self, nsamples, ndims):
        return np.random.uniform(-1., 1., size=[nsamples, ndims])

    def _sample_real_x(self, data_size, batch_size):
        return np.random.permutation(data_size)[:batch_size]

    def _encode_data(self, data):
        n_samples = len(data)
        features_encoded = np.empty((n_samples, self.nfeatures))
        cidx = 0

        for attr_name, cdict in self.metadata.items():
            data_type = cdict['type']
            col_data = data[attr_name].to_numpy()

            if data_type == FLOAT or data_type == INTEGER:
                # Normalise continuous data
                if self.infer_ranges:
                    col_max = max(col_data)
                    col_min = min(col_data)

                    self.metadata[attr_name]['max'] = col_max
                    self.metadata[attr_name]['min'] = col_min

                else:
                    col_max = cdict['max']
                    col_min = cdict['min']

                features_encoded[:, cidx] = np.true_divide(col_data - col_min, col_max + ZERO_TOL)

                cidx += 1

            elif data_type == CATEGORICAL or data_type == ORDINAL:
                # One-hot encoded categorical columns
                col_cats = cdict['categories']
                col_data_onehot = self._one_hot(col_data, col_cats)
                features_encoded[:, cidx : cidx + len(col_cats)] = col_data_onehot

                cidx += len(col_cats)

        return features_encoded

    def _decode_data(self, features_encoded):
        """ Revers feature encoding. """
        data = DataFrame(columns=self.attribute_list)

        cidx = 0

        for attr_name, cdict in self.metadata.items():
            data_type = cdict['type']

            if data_type == FLOAT:
                col_min = cdict['min']
                col_max = cdict['max']

                col_data = features_encoded[:, cidx]
                col_data = col_data * (col_max + ZERO_TOL) + col_min
                data[attr_name] = col_data.astype(float)
                cidx += 1

            elif data_type == INTEGER:
                col_min = cdict['min']
                col_max = cdict['max']

                col_data = features_encoded[:, cidx]
                col_data = col_data * (col_max + ZERO_TOL) + col_min
                data[attr_name] = col_data.astype(int)
                cidx += 1

            elif data_type == CATEGORICAL or data_type == ORDINAL:
                col_cats = cdict['categories']
                ncats = len(col_cats)

                col_data_onehot = features_encoded[:, cidx : cidx + ncats]
                col_data = self._reverse_one_hot(col_data_onehot, col_cats)
                data[attr_name] = col_data.astype(str)

                cidx += ncats

        return data

    def _one_hot(self, col_data, categories):
        col_data_onehot = np.zeros((len(col_data), len(categories)))
        cidx = [categories.index(c) for c in col_data]
        col_data_onehot[np.arange(len(col_data)), cidx] = 1

        return col_data_onehot

    def _reverse_one_hot(self, col_encoded, categories):
        cat_idx = np.argmax(col_encoded, axis=1)
        col_data = np.array([categories[i] for i in cat_idx])

        return col_data