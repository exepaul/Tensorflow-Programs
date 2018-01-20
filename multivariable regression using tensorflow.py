#input variables

a=[[1,2,3,4,5]]
b=[[2,4,6,8,10]]
c=[[3,6,9,12,15]]

#prediction aim


d=[[5,10,15,20,25]]



import tensorflow as tf
tf.set_random_seed(777)

var_a=tf.Variable(tf.random_normal([1]))
var_b=tf.Variable(tf.random_normal([1]))
var_c=tf.Variable(tf.random_normal([1]))
var_d=tf.Variable(tf.random_normal([1]))

pla_x=tf.placeholder(tf.float32,shape=[None,5])
pla_y=tf.placeholder(tf.float32,shape=[None,5])
pla_z=tf.placeholder(tf.float32,shape=[None,5])
pla_zz=tf.placeholder(tf.float32,shape=[None,5])

hy=pla_x*var_a+pla_y*var_b+pla_z*var_c+var_d
cost=tf.reduce_mean(tf.square(hy-pla_zz))
train=tf.train.GradientDescentOptimizer(0.001).minimize(cost)
var_ini=tf.global_variables_initializer()

with tf.Session() as tt:
    tt.run(var_ini)
    for i in range(2000):
        aa,ba,ca,da,daa,ff,to=tt.run([var_a,var_b,var_c,var_d,hy,cost,train],feed_dict={pla_x:a,pla_y:b,pla_z:c,pla_zz:d})
        print(aa,ba,ca)
        
        
        
 # 
# [ 2.9608078] [ 0.68385226] [ 0.20058268]
# [ 2.96080971] [ 0.68385583] [ 0.20058808]
# [ 2.96081161] [ 0.68385941] [ 0.20059347]
# [ 2.96081352] [ 0.68386298] [ 0.20059885]
# [ 2.96081543] [ 0.68386656] [ 0.20060423]
# [ 2.96081734] [ 0.68387014] [ 0.20060961]
# [ 2.96081924] [ 0.68387371] [ 0.20061497]
# [ 2.96082115] [ 0.68387729] [ 0.20062034]
# [ 2.96082306] [ 0.68388087] [ 0.20062572]
# [ 2.96082497] [ 0.68388444] [ 0.20063108]
# [ 2.96082664] [ 0.68388802] [ 0.20063643]




#lets check the result :

x1=[1,2,3,4,5,6,7]
x2=[2,4,6,8,10,12,14]
x3=[3,6,9,12,15,18,21]
y=[5,10,15,20,25,30]


data=[2.96199965,0.68624687, 0.20417139,0.22942682]





for i,j,k in zip(x1,x2,x3):
    print(i*data[0]+j*data[1]+k*data[2]+data[3])

#  5.17643438                      5
# 10.123441940000001               10
# 15.070449500000002               15
# 20.01745706                      20
# 24.96446462                      25
# 29.911472180000004               30
# 34.85847974                      #prediction


    
