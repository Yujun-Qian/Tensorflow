import tensorflow as tf

#parameters for the net
w1 = tf.Variable(tf.random_uniform(shape=[2,2], minval=-1, maxval=1, name='weights1'))
w2 = tf.Variable(tf.random_uniform(shape=[2,1], minval=-1, maxval=1, name='weights2'))

#biases
b1 = tf.Variable(tf.zeros([2]), name='bias1')
b2 = tf.Variable(tf.zeros([1]), name='bias2')

#tensorflow session
sess = tf.Session()


def train():

    #placeholders for the traning inputs (4 inputs with 2 features each) and outputs (4 outputs which have a value of 0 or 1)
    x = tf.placeholder(tf.float32, [4, 2], name='x-inputs')
    y = tf.placeholder(tf.float32, [4, 1], name='y-inputs')

    #set up the model calculations
    temp = tf.sigmoid(tf.matmul(x, w1) + b1)
    output = tf.sigmoid(tf.matmul(temp, w2) + b2)

    #cost function is avg error over training samples
    cost = tf.reduce_mean(((y * tf.log(output)) + ((1 - y) * tf.log(1.0 - output))) * -1)

    #training step is gradient descent
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

    #declare training data
    training_x = [[0,1], [0,0], [1,0], [1,1]]
    training_y = [[1], [0], [1], [0]]

    #init session
    init = tf.initialize_all_variables()
    sess.run(init)

    #training
    for i in range(100000):
        sess.run(train_step, feed_dict={x:training_x, y:training_y})

        if i % 1000 == 0:
            print (i, sess.run(cost, feed_dict={x:training_x, y:training_y}))

    print '\ntraining done\n'


def test(inputs):
    #redefine the shape of the input to a single unit with 2 features
    xtest = tf.placeholder(tf.float32, [1, 2], name='x-inputs')

    #redefine the model in terms of that new input shape
    temp = tf.sigmoid(tf.matmul(xtest, w1) + b1)
    output = tf.sigmoid(tf.matmul(temp, w2) + b2)

    print (inputs, sess.run(output, feed_dict={xtest:[inputs]})[0, 0])

train()

test([0,1])
test([0,0])
test([1,1])
test([1,0])
