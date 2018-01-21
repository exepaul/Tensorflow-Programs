import tensorflow as tf
import numpy as np

#data_cleaning and data_loading

data_1=[]
final_data=[]
prediction=[]

with open('mp_data.csv','r') as f:
    for line in f:
        if line.split()[3]!='?':
            data_1.append(line.split())

for i in data_1:
    prediction.append([float(i[0])])
    final_data.append([float(i[3]),float(i[4]),float(i[5])])

print(final_data)
print(prediction)




tf.set_random_seed(777)

var_a=tf.Variable(tf.random_normal([3,1]))
var_b=tf.Variable(tf.random_normal([1]))

pla_x=tf.placeholder(tf.float32,shape=[None,3])
pla_y=tf.placeholder(tf.float32,shape=[None,1])

hypothesis=tf.matmul(pla_x,var_a) + var_b

cost=tf.reduce_mean(tf.square(hypothesis-pla_y))

train=tf.train.GradientDescentOptimizer(0.00000001).minimize(cost)

var_ini=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(var_ini)
    for i in range(1000000):
        a1,a2,a3,a4,a5=sess.run([var_a,var_b,hypothesis,cost,train],feed_dict={pla_x:final_data,pla_y:prediction})
        print(a1,a2)
        print("cost",a4)
