import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Logistic,RelaxedBernoulli,BernoulliWithSigmoidProbs

def A_tf(f):
    """
    Computes the log partition function A
    
    Inputs:
        f:  Real valued tensor of shape [batch size,number of neurons], is the "activation" 
            of each neuron for each sample
    Ouputs:
        A:  Real valued tensor of shape [batch size,number of neurons], is the "log partition" 
            of each neuron for each sample.  Will need to be summed along axis 1 to get population log partition
    """
    return tf.abs(f) + tf.nn.softplus(-tf.abs(2.0*f))

def I_hat(f,temperature,samples_per_x):
    """
    Computes a stochastic approximation to the mutual information using RelaxedBernoulli sampling 
    
    Inputs:
        f:  A real valued tensor of shape [batch_size,number_of_neurons].  f[i,j] is the "activation" of the jth neuron for sample i.
        temperature:  An 0-D Tensor, representing the temperature of the RelaxedBernoulli distributions. The temperature should be
            strictly greater than 0.
        samples_per_x:  A 0-D tensor of type int32 or int64.  How many samples of \vec{r} to draw per sample of x.
    Outputs:
        I:  Real valued scalar.  Stochastic estimate of the mutual information.
        r_samples:  Tensor of shape [samples_per_x*batch_size,number_of_neurons]. r_samples[i*samples_per_x:(i+1)*samples_per_x,j] are
            samples for the jth neuron for sample i.
    """
    A = tf.reduce_sum(A_tf(f),1)

    mean_r = tf.tanh(f)
    term1 = tf.reduce_mean(tf.reduce_sum(mean_r*f,1),0) - tf.reduce_mean(A,0)

    logits = 2*tf.tile(f,[samples_per_x,1])
    q = RelaxedBernoulli(temperature,logits=logits)
    r_samples = 2.0*q.sample() - 1.0 #r_samples is shape (batch_size*samples_per_x,N)
    
    C = tf.tensordot(r_samples,f,axes=([1],[1])) - tf.tile(tf.expand_dims(A,0),[tf.shape(r_samples)[0],1])
    C_alt = tf.reduce_logsumexp(C,1) - tf.log(tf.to_float(tf.shape(f)[0])) #better than C_2
    term2 = tf.reduce_mean(C_alt,0)
    I = term1 - term2
    return I,r_samples