
import tensorflow as tf
tf.set_random_seed(777)
sess=tf.InteractiveSession()
import numpy as np

print(np.array([[1,2,3,4,5],[1,2,3,4,5]]).shape)
pla_x=tf.placeholder(tf.float32,shape=[None,5])
pla_y=tf.placeholder(tf.float32,shape=[5,None])


w=pla_x*12
w_1=pla_y*12
print(sess.run(w,feed_dict={pla_x:np.array([[1,2,3,4,5]])}))
print(sess.run(w_1,feed_dict={pla_y:[[1.1],[2.1],[3.1],[4.1],[5.]]}))



#result
(2, 5)
[[ 12.  24.  36.  48.  60.]]
[[ 13.20000076]
 [ 25.19999886]
 [ 37.19999695]
 [ 49.19999695]
 [ 60.        ]]
