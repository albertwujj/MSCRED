import tensorflow as tf
import numpy as np
from mscred import MSCRED

sigs = np.random.randn(25,30,30,3)

input_ph = tf.placeholder(tf.float32, shape=(25,30,30,3))
model = MSCRED()
residual = model(input_ph)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
resid_val = sess.run([residual], feed_dict={input_ph: sigs})
print(resid_val)
