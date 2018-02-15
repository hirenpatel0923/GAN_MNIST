import tensorflow as tf

w = tf.Variable(0.1)
x = tf.Variable(2.5)
a = x*w

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(a)
sw = tf.summary.FileWriter('opt', sess.graph)
sw.close()