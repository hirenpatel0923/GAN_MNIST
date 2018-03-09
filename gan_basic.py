from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import xavier_initializer

mnist = input_data.read_data_sets('MNIST_data')

#descrimator net
D_W1 = tf.get_variable(shape=[784, 128], name='D_W1', initializer=xavier_initializer())
D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')

D_W2 = tf.get_variable(shape=[128, 1], name='D_W2', initializer=xavier_initializer())
D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')

theta_D = [D_W1, D_W2, D_b1, D_b2]

#generator net
G_W1 = tf.get_variable(shape=[100, 128], name='G_W1', initializer=xavier_initializer())
G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')

G_W2 = tf.get_variable(shape=[128, 784], name='G_W2', initializer=xavier_initializer())
G_b2 = tf.Variable(tf.zeros(shape=[784]), name='G_b2')

theta_G = [G_W1, G_W2, G_b1, G_b2]

z_dim = 100
mb_size = 50
Z = tf.placeholder('float', shape=[None, z_dim], name='Z')
X = tf.placeholder('float', shape=[None, 784], name='X')

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob

def descriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit

G_sample = generator(Z)
D_real, D_logit_real = descriminator(X)
D_fake, D_logit_fake = descriminator(G_sample)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

def sample_Z(m,n):
    return np.random.uniform(-1., 1., size=[m,n])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for it in range(100000):
    X_mb, _ = mnist.train.next_batch(mb_size)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z:sample_Z(mb_size, z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, z_dim)})

    if(it % 1000 == 0):
        print('it -> ', it)
    if(it % 10000 == 0):
        with tf.variable_scope('gen', reuse=True):
            generated_images = generator(Z)

        images = sess.run(generated_images, {Z: sample_Z(mb_size, z_dim)})
        plt.imshow(images[0].reshape([28, 28]), cmap='Greys')
        plt.show()


