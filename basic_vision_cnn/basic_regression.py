import tensorflow as tf
import numpy as np


def main():
    trainX = np.linspace(-1,1,101)
    trainY = 3*trainX + np.random.randn(*trainX.shape)*0.33

    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    w = tf.Variable(0.0,name="weights")
    yModel = tf.multiply(X,w)

    cost = tf.pow(Y-yModel,2)

    trainOp = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    initOp = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(initOp)
        for i in range(100):
            for (x,y) in zip(trainX, trainY):
                sess.run(trainOp,feed_dict = {X:x, Y:y})
        print(sess.run(w))




if __name__ == "__main__":
    main()

