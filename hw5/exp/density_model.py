import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from ex_utils import build_mlp

class Density_Model(object):
    def __init__(self):
        super(Density_Model, self).__init__()

    def receive_tf_sess(self, sess):
        self.sess = sess

    def get_prob(self, state):
        raise NotImplementedError

class Histogram(Density_Model):
    def __init__(self, nbins, preprocessor):
        super(Histogram, self).__init__()
        self.nbins = nbins
        self.total = 0.
        self.hist = {}
        for i in range(int(self.nbins)):
            self.hist[i] = 0
        self.preprocessor = preprocessor

    def update_count(self, state, increment):
        """
            ### PROBLEM 1
            ### YOUR CODE HERE

            args:
                state: numpy array
                increment: int

            TODO:
                1. increment the entry "bin_name" in self.hist by "increment"
                2. increment self.total by "increment" 
        """
        bin_name = self.preprocessor(state)
        self.hist[bin_name] += increment
        self.total += increment
        #raise NotImplementedError

    def get_count(self, states):
        """
            ### PROBLEM 1
            ### YOUR CODE HERE

            args:
                states: numpy array (bsize, ob_dim)

            returns: 
                counts: numpy_array (bsize)

            TODO:
                For each state in states:
                    1. get the bin_name using self.preprocessor
                    2. get the value of self.hist with key bin_name
        """
        # print(states.shape)
        bsize,ob_dim = states.shape

        counts = []
        for i in range(bsize):            
            bin_name = self.preprocessor(states[i,:])
            counts.append(self.hist[bin_name])
        counts = np.array(counts)
        return counts

    def get_prob(self, states):
        """
            ### PROBLEM 1
            ### YOUR CODE HERE

            args:
                states: numpy array (bsize, ob_dim)
    
            returns:
                return the probabilities of the state (bsize)

            NOTE:
                remember to normalize by float(self.total)
        """
        bsize,ob_dim = states.shape
        
        probs = []
        for i in range(bsize):            
            bin_name = self.preprocessor(states[i,:])
            count = self.hist[bin_name]
            probs.append(count/self.total)
        probs = np.array(probs)
        # raise NotImplementedError
        return probs

class RBF(Density_Model):
    """
        https://en.wikipedia.org/wiki/Radial_basis_function_kernel
        https://en.wikipedia.org/wiki/Kernel_density_estimation
    """
    def __init__(self, sigma):
        super(RBF, self).__init__()
        self.sigma = sigma
        self.means = None

    def fit_data(self, data):
        """
            ### PROBLEM 2
            ### YOUR CODE HERE

            args:
                data: list of states of shape (ob_dim)

            TODO:
                We simply assign self.means to be equal to the data points.
                Let the length of the data be B
                self.means: np array (B, ob_dim)
        """
        B, ob_dim = len(data), len(data[0])
        # raise NotImplementedError
        self.means = np.zeros((B,ob_dim))
        for i in range(B):
            self.means[i,:] = data[i]

        # self.means = None
        assert self.means.shape == (B, ob_dim)

    def get_prob(self, states):
        """
            ### PROBLEM 2
            ### YOUR CODE HERE

            given:
                states: (b, ob_dim)
                    where b is the number of states we wish to get the
                    probability of

                self.means: (B, ob_dim)
                    where B is the number of states in the replay buffer
                    we will plop a Gaussian distribution on top of each
                    of self.means with a std of self.sigma

            TODO:
                1. Compute deltas: for each state in states, compute the
                    difference between that state and every mean in self.means.
                2. Euclidean distance: sum the squared deltas
                3. Gaussian: evaluate the probability of the state under the 
                    gaussian centered around each mean. The hyperparameters
                    for the reference solution assume that you do not normalize
                    the gaussian. This is fine since the rewards will be 
                    normalized later when we compute advantages anyways.
                4. Average: average the probabilities from each gaussian
        """
        b, ob_dim = states.shape
        if self.means is None:
            # Return a uniform distribution if we don't have samples in the 
            # replay buffer yet.
            return (1.0/len(states))*np.ones(len(states))
        else:
            B, replay_dim = self.means.shape
            assert states.ndim == self.means.ndim and ob_dim == replay_dim

            # 1. Compute deltas
            deltas = np.zeros((b, B, ob_dim))
            for i1 in range(b):
                state_b = states[i1,:]
                for i2 in range(B):
                    deltas[i1,i2,:] = state_b - self.means[i2,:]

            assert deltas.shape == (b, B, ob_dim)

            euc_dists = np.zeros((b,B))
            for i1 in range(b):
                for i2 in range(B):
                    delt = np.sum(np.square(deltas[i1,i2,:]))
                    euc_dists[i1,i2] = delt

            # 2. Euclidean distance
            # euc_dists = raise NotImplementedError
            assert euc_dists.shape == (b, B)

            gaussians = np.exp((-1*euc_dists)/(2*np.square(self.sigma)))
            # Gaussian-
            # gaussians = raise NotImplementedError
            assert gaussians.shape == (b, B)
            densities = np.sum(gaussians,axis = 1)/B
            # 4. Average
            # densities = raise NotImplementedError
            assert densities.shape == (b,)

            return densities

class Exemplar(Density_Model):
    def __init__(self, ob_dim, hid_dim, learning_rate, kl_weight):
        super(Exemplar, self).__init__()
        self.ob_dim = ob_dim
        self.hid_dim = hid_dim
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight

    def build_computation_graph(self):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE

            TODO:
                1. self.log_likelihood. shape: (batch_size)
                    - use tf.squeeze
                    - use the discriminator to get the log prob of the discrim_target
                2. self.likelihood. shape: (batch_size)
                    - use tf.squeeze
                    - use the discriminator to get the prob of the discrim_target
                3. self.kl. shape: (batch_size)
                    - simply add the kl divergence between self.encoder1 and 
                        the prior and the kl divergence between self.encoder2 
                        and the prior. Do not average.
                4. self.elbo: 
                    - subtract the kl (weighted by self.kl_weight) from the 
                        log_likelihood, and average over the batch
                5. self.update_op: use the AdamOptimizer with self.learning_rate 
                    to minimize the -self.elbo (Note the negative sign!)

            Hint:
                https://www.tensorflow.org/probability/api_docs/python/tfp/distributions
        """
        self.state1, self.state2 = self.define_placeholders()
        self.encoder1, self.encoder2, self.prior, self.discriminator = self.forward_pass(self.state1, self.state2)
        self.discrim_target = tf.placeholder(shape=[None, 1], name="discrim_target", dtype=tf.float32)

        # raise NotImplementedError
        self.log_likelihood = tf.squeeze(self.discriminator.log_prob(self.discrim_target,name='log_prob'),axis = 1)
        self.likelihood = tf.squeeze(self.discriminator.prob(self.discrim_target,name='prob'),axis = 1)
        print(11,11,tf.shape(self.likelihood))
        self.kl = tf.distributions.kl_divergence(self.encoder1,self.prior) + tf.distributions.kl_divergence(self.encoder2,self.prior) 
        # print(len(self.log_likelihood.shape))
        # print(len(self.likelihood.shape))
        # print(len(self.kl.shape))
        assert len(self.log_likelihood.shape) == len(self.likelihood.shape) == len(self.kl.shape) == 1## Shape of the placeholder?

        # raise NotImplementedError
        self.elbo = tf.reduce_mean(self.log_likelihood - self.kl_weight * self.kl)
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss=-self.elbo)

    def define_placeholders(self):
        state1 = tf.placeholder(shape=[None, self.ob_dim], name="s1", dtype=tf.float32)
        state2 = tf.placeholder(shape=[None, self.ob_dim], name="s2", dtype=tf.float32)
        return state1, state2

    def make_encoder(self, state, z_size, scope, n_layers, hid_size):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE

            args:
                state: tf variable
                z_size: output dimension of the encoder network
                scope: scope name
                n_layers: number of layers of the encoder network
                hid_size: hidden dimension of encoder network

            TODO:
                1. z_mean: the output of a neural network that takes the state as input,
                    has output dimension z_size, n_layers layers, and hidden 
                    dimension hid_size
                2. z_logstd: a trainable variable, initialized to 0
                    shape (z_size,)

            Hint: use build_mlp
        """
        z_mean = build_mlp(state, z_size, scope, n_layers, hid_size, activation=tf.tanh, output_activation=None)
        z_logstd = tf.get_variable("logstd",shape=[z_size]) 
        # z_mean = raise NotImplementedError
        # z_logstd = raise NotImplementedError
        return tfp.distributions.MultivariateNormalDiag(loc=z_mean, scale_diag=tf.exp(z_logstd))

    def make_prior(self, z_size):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE

            args:
                z_size: output dimension of the encoder network

            TODO:
                prior_mean and prior_logstd are for a standard normal distribution
                    both have dimension z_size
        """
        prior_mean = tf.zeros([z_size])
        prior_logstd = tf.zeros([z_size])
        return tfp.distributions.MultivariateNormalDiag(loc=prior_mean, scale_diag=tf.exp(prior_logstd))

    def make_discriminator(self, z, output_size, scope, n_layers, hid_size):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE

            args:
                z: input to to discriminator network
                output_size: output dimension of discriminator network
                scope: scope name
                n_layers: number of layers of discriminator network
                hid_size: hidden dimension of discriminator network 

            TODO:
                1. logit: the output of a neural network that takes z as input,
                    has output size output_size, n_layers layers, and hidden
                    dimension hid_size

            Hint: use build_mlp
        """
        logit = build_mlp(z, output_size, scope, n_layers, hid_size, activation=tf.tanh, output_activation=None)
        # logit = raise NotImplementedError
        return tfp.distributions.Bernoulli(logit)

    def forward_pass(self, state1, state2):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE

            args:
                state1: tf variable
                state2: tf variable
            
            encoder1: tfp.distributions.MultivariateNormalDiag distribution
            encoder2: tfp.distributions.MultivariateNormalDiag distribution
            prior: tfp.distributions.MultivariateNormalDiag distribution
            discriminator: tfp.distributions.Bernoulli distribution

            TODO:
                1. z1: sample from encoder1
                2. z2: sample from encoder2
                3. z: concatenate z1 and z2

            Hint: 
                https://www.tensorflow.org/probability/api_docs/python/tfp/distributions
        """
        # Reuse
        make_encoder1 = tf.make_template('encoder1', self.make_encoder)
        make_encoder2 = tf.make_template('encoder2', self.make_encoder)
        make_discriminator = tf.make_template('decoder', self.make_discriminator)

        # Encoder
        encoder1 = make_encoder1(state1, self.hid_dim/2, 'z1', n_layers=2, hid_size=self.hid_dim)
        encoder2 = make_encoder2(state2, self.hid_dim/2, 'z2', n_layers=2, hid_size=self.hid_dim)

        # Prior
        prior = self.make_prior(self.hid_dim/2)

        # Sampled Latent
        z1 = encoder1.sample()
        z2 = encoder2.sample()
        z = tf.concat([z1,z2],axis=1)
        # z2 = raise NotImplementedError
        # z = raise NotImplementedError
        print(tf.shape(z))
        # Discriminator
        discriminator = make_discriminator(z, 1, 'discriminator', n_layers=2, hid_size=self.hid_dim)
        return encoder1, encoder2, prior, discriminator

    def update(self, state1, state2, target):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE

            args:
                state1: np array (batch_size, ob_dim)
                state2: np array (batch_size, ob_dim)
                target: np array (batch_size, 1)

            TODO:
                train the density model and return
                    ll: log_likelihood
                    kl: kl divergence
                    elbo: elbo
        """
        assert state1.ndim == state2.ndim == target.ndim
        assert state1.shape[1] == state2.shape[1] == self.ob_dim
        assert state1.shape[0] == state2.shape[0] == target.shape[0]

        _,elbo,ll,kl = self.sess.run([self.update_op, self.elbo,self.log_likelihood,self.kl], feed_dict={self.state1: state1,self.state2: state2,self.discrim_target: target})

        # raise NotImplementedError
        return ll, kl, elbo

    def get_likelihood(self, state1, state2):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE

            args:
                state1: np array (batch_size, ob_dim)
                state2: np array (batch_size, ob_dim)

            TODO:
                likelihood of state1 == state2

            Hint:
                what should be the value of self.discrim_target?
        """
        assert state1.ndim == state2.ndim
        assert state1.shape[1] == state2.shape[1] == self.ob_dim
        assert state1.shape[0] == state2.shape[0]
        # raise NotImplementedError

        batch_size,ob_dim = state1.shape
        likelihood = np.zeros((batch_size,1))
        for i in range(batch_size):
            if state1[i,:].all() == state2[i,:].all():
                likelihood[i,0] = 1

        print(likelihood)

        likelihood_1 = self.sess.run(self.likelihood,feed_dict = {self.state1:state1,self.state2:state2,self.discrim_target:likelihood})
        # print(2,likelihood_1.shape)
        return likelihood_1

    def get_prob(self, state):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE
        
            args:
                state: np array (batch_size, ob_dim)

            TODO:
                likelihood: 
                    evaluate the discriminator D(x,x) on the same input
                prob:
                    compute the probability density of x from the discriminator
                    likelihood (see homework doc)
        """
        # likelihood = raise NotImplementedError
        # avoid divide by 0 and log(0)
        likelihood = self.get_likelihood(state,state)
        print(np.squeeze(likelihood).shape)
        likelihood = np.clip(np.squeeze(likelihood), 1e-5, 1-1e-5)
        # print(likelihood.shape)
        prob = (1-likelihood)/likelihood
        # prob = raise NotImplementedError
        return prob
