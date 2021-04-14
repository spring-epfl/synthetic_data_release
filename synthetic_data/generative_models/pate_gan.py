"""
A generative model training algorithm based on
"PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees"
by J. Yoon, J. Jordon, M. van der Schaar, published in International Conference on Learning Representations (ICLR), 2019

Adapted from: https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/82d7f91d46db54d256ff4fc920d513499ddd2ab8/alg/pategan/
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import pandas as pd
from tqdm import tqdm

from .generative_model import GenerativeModel


MB_SIZE = 128
C_DIM = 1
LAM = 10
LR = 1e-4

NITER = 100
NUM_TEACHERS = 10


class PateGan(GenerativeModel):
    """ A generative adversarial network trained under the PATE framework to achieve differential privacy """

    def __init__(self, metadata, eps, delta):
        """

        :param metadata: dict: Attribute metadata describing the data domain of the synthetic target data
        :param eps: float: Privacy parameter
        :param delta: float: Privacy parameter
        """

        self.metadata = metadata
        self.epsilon = eps
        self.delta = delta
        self.datatype = pd.DataFrame

        self.trained = False

        self.__name__ = f'PateGan{self.epsilon}'

    def fit(self, data):
        """Fit the generative model of the training data distribution.

        :param data: DataFrame: Training set
        """

        X_train, Y_train, cols_to_reverse = self._one_hot(data)

        self.columns_to_reverse = cols_to_reverse

        self.no, self.X_dim = X_train.shape
        self.z_dim = int(self.X_dim / 4)
        self.h_dim = int(self.X_dim)

        # Feature matrix
        self.X = tf.placeholder(tf.float32, shape=[None, self.X_dim])
        # Target variable
        self.Y = tf.placeholder(tf.float32, shape=[None, C_DIM])
        # Latent space
        self.Z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        # Conditional variable
        self.M = tf.placeholder(tf.float32, shape=[None, C_DIM])
        self.Y_train = Y_train

        lamda = np.sqrt(2 * np.log(1.25 * (10 ^ (self.delta)))) / self.epsilon

        # Data Preprocessing
        X_train = np.asarray(X_train)
        self.Min_Val = np.min(X_train, 0)
        X_train = X_train - self.Min_Val
        self.Max_Val = np.max(X_train, 0)
        X_train = X_train / (self.Max_Val + 1e-8)
        self.dim = len(X_train[:,0])

        # Generator
        self.G_sample = self._generator(self.Z,self.Y)

        # Discriminator
        D_real = self._discriminator(self.X, self.Y)
        D_fake = self._discriminator(self.G_sample, self.Y)
        D_entire = tf.concat(axis=0, values=[D_real, D_fake])

        # Replacement of Clipping algorithm to Penalty term
        # 1. Line 6 in Algorithm 1
        eps = tf.random_uniform([MB_SIZE, 1], minval=0., maxval=1.)
        X_inter = eps * self.X + (1. - eps) * self.G_sample

        # 2. Line 7 in Algorithm 1
        grad = tf.gradients(self._discriminator(X_inter, self.Y), [X_inter, self.Y])[0]
        grad_norm = tf.sqrt(tf.reduce_sum((grad) ** 2 + 1e-8, axis=1))
        grad_pen = LAM * tf.reduce_mean((grad_norm - 1) ** 2)

        # Loss function
        D_loss = tf.reduce_mean((1 - self.M) * D_entire) - tf.reduce_mean(self.M * D_entire) + grad_pen
        G_loss = -tf.reduce_mean(D_fake)

        # Solver
        D_solver = (tf.train.AdamOptimizer(learning_rate=LR, beta1=0.5).minimize(D_loss, var_list=self.theta_D))
        G_solver = (tf.train.AdamOptimizer(learning_rate=LR, beta1=0.5).minimize(G_loss, var_list=self.theta_G))

        # Start session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # Training iterations
        for _ in tqdm(range(NITER)):
            for _ in range(NUM_TEACHERS):
                # Teacher training
                Z_mb = self._sample_Z(MB_SIZE, self.z_dim)

                # Teacher 1
                X_idx = self._sample_X(self.no, MB_SIZE)
                X_mb = X_train[X_idx, :]

                Y_mb = np.reshape(Y_train[X_idx], [MB_SIZE, 1])

                M_real = np.ones([MB_SIZE, ])
                M_fake = np.zeros([MB_SIZE, ])

                M_entire = np.concatenate((M_real, M_fake), 0)

                Normal_Add = np.random.normal(loc=0.0, scale=lamda, size=MB_SIZE * 2)

                M_entire = M_entire + Normal_Add

                M_entire = (M_entire > 0.5)

                M_mb = np.reshape(M_entire.astype(float), (2 * MB_SIZE, 1))

                _, D_loss_curr = self.sess.run([D_solver, D_loss], feed_dict={self.X: X_mb, self.Z: Z_mb, self.M: M_mb, self.Y: Y_mb})

            # Generator Training
            Z_mb = self._sample_Z(MB_SIZE, self.z_dim)

            X_idx = self._sample_X(self.no, MB_SIZE)

            Y_mb = np.reshape(Y_train[X_idx], [MB_SIZE, 1])

            _, G_loss_curr = self.sess.run([G_solver, G_loss], feed_dict={self.Z: Z_mb, self.Y: Y_mb})

        self.trained = True

    def generate_samples(self, nsamples):
        """""
        Samples synthetic data records from the fitted generative distribution

        :param nsamples: int: Number of synthetic records to generate
        :return synData: DataFrame: A synthetic dataset
        """
        # Output generation
        New_X_train = self.sess.run([self.G_sample], feed_dict={self.Z: self._sample_Z(self.dim, self.z_dim),
                                                           self.Y: np.reshape(self.Y_train, [len(self.Y_train), 1])})

        New_X_train = New_X_train[0]

        # Renormalization
        New_X_train = New_X_train * (self.Max_Val + 1e-8)
        New_X_train = New_X_train + self.Min_Val
        New_X_train = np.concatenate((New_X_train,np.reshape(self.Y_train, [len(self.Y_train), 1])), axis = 1)
        np.random.shuffle(New_X_train)

        return self._reverse_one_hot(New_X_train[:nsamples])

    def _generator(self, z, y):
        """
        PateGan generator implementation

        :param z: training data
        :param y: training labels
        """
        G_W1 = tf.Variable(self._xavier_init([self.z_dim + C_DIM, self.h_dim]))
        G_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]))

        G_W2 = tf.Variable(self._xavier_init([self.h_dim, self.h_dim]))
        G_b2 = tf.Variable(tf.zeros(shape=[self.h_dim]))

        G_W3 = tf.Variable(self._xavier_init([self.h_dim, self.X_dim]))
        G_b3 = tf.Variable(tf.zeros(shape=[self.X_dim]))

        self.theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

        inputs = tf.concat([z, y], axis=1)
        G_h1 = tf.nn.tanh(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.tanh(tf.matmul(G_h1, G_W2) + G_b2)
        G_log_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)

        return G_log_prob

    def _discriminator(self, x, y):
        """
        PateGan generator implementation

        :param x: training data
        :param y: training labels
        """

        D_W1 = tf.Variable(self._xavier_init([self.X_dim + C_DIM, self.h_dim]))
        D_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]))

        D_W2 = tf.Variable(self._xavier_init([self.h_dim, self.h_dim]))
        D_b2 = tf.Variable(tf.zeros(shape=[self.h_dim]))

        D_W3 = tf.Variable(self._xavier_init([self.h_dim, 1]))
        D_b3 = tf.Variable(tf.zeros(shape=[1]))

        self.theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
        inputs = tf.concat([x, y], axis=1)
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        out = (tf.matmul(D_h2, D_W3) + D_b3)

        return out

    def _xavier_init(self,size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)

        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def _sample_Z(self,m, n):
        return np.random.uniform(-1., 1., size=[m, n])

    def _sample_X(self,m, n):
        return np.random.permutation(m)[:n]

    def _one_hot(self, data):
        continuous_columns = self.metadata['continuous_columns']
        categorical_columns = sorted(self.metadata['categorical_columns'] + self.metadata['ordinal_columns'])
        if data is pd.DataFrame():
            df = data

        else:
            df = pd.DataFrame(data)

        columns_to_reverse = []
        names = []
        cat_attr_names = []
        for cidx in range(df.shape[1]-1):
            if cidx in categorical_columns:
                col = self.metadata['columns'][cidx]
                cat_attr_names.append(col['i2s'])
                columns_to_reverse.append(col['i2s'])
                name = col['name']
                names.append(name)
                data = df[name].tolist()
                oh_d = self._col_one_hot(col['i2s'], data)
                if cidx == 0:
                    oh_data = np.array(oh_d)
                else:

                    oh_data = np.concatenate((oh_data, oh_d), axis = 1)
            elif cidx in continuous_columns:
                col = self.metadata['columns'][cidx]
                name = col['name']
                names.append(name)
                data = df[name].to_numpy()
                max = col['max']
                columns_to_reverse.append([max])

                oh_d = np.true_divide(data, max)
                if cidx == 0:
                    oh_data =np.array(oh_d).reshape((data.shape[0],-1))
                else:
                    oh_data = np.concatenate((oh_data, np.array(oh_d).reshape((data.shape[0],-1))), axis = 1)

        cidx += 1

        col = self.metadata['columns'][cidx]
        name = col['name']
        names.append(name)
        data = df[name].tolist()
        y = [col['i2s'].index(data[i]) for i in range(len(data))]

        self.col_names = names

        return oh_data, np.array(y), columns_to_reverse

    def _col_one_hot(self, col_options, data):
        attr = len(col_options)

        oh_data = np.zeros((len(data),attr))
        for i in range(len(data)):
            oh_data[i, col_options.index(data[i]) ] = 1

        return oh_data

    def _reverse_one_hot(self, data):

        data2 = data
        i=0
        for item in self.columns_to_reverse:

            d1 = data2[:, i:i + len(item)]
            if len(item) == 1:
                r_data = d1*item[0]
                r_data = r_data.astype(str).astype(float)
            else:
                r_data = np.argmax(d1, axis=1)
                r_data = np.array([item[x] for x in r_data])


            if i == 0:
                recovered_data = r_data.transpose().reshape(-1, 1)
                i += len(item)
            else:
                recovered_data = np.concatenate((recovered_data, r_data.transpose().reshape(-1, 1)), axis=1)
                i += len(item)
        continous_cols = self.metadata['continuous_columns']
        recovered_data =  np.concatenate((recovered_data, data[:,-1].reshape(-1, 1)), axis=1)
        recovered_data = pd.DataFrame(recovered_data, columns = self.col_names)
        for cidx  in continous_cols:
            col = self.metadata['columns'][cidx]
            name = col['name']
            recovered_data[name] = pd.to_numeric(recovered_data[name])

        return recovered_data
