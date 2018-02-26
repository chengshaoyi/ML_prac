import tensorflow as tf
import dataset

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))





def prepareTrainingData(validation_size, img_size, train_path, classes):
    data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
    return data


def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters):
    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    layer = tf.nn.relu(layer)

    layer = tf.nn.max_pool(value=layer,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')

    return layer


def create_flatten_layer(layer):
    # We know that the shape of the layer will be [batch_size img_size img_size num_channels]
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    # Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def buildCnn(x, num_channels, num_classes,
             filter_size_conv1, num_filters_conv1,
             filter_size_conv2, num_filters_conv2,
             filter_size_conv3, num_filters_conv3,
             fc_layer_size):
    layer_conv1 = create_convolutional_layer(input=x,
                                             num_input_channels=num_channels,
                                             conv_filter_size=filter_size_conv1,
                                             num_filters=num_filters_conv1)
    layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                             num_input_channels=num_filters_conv1,
                                             conv_filter_size=filter_size_conv2,
                                             num_filters=num_filters_conv2)

    layer_conv3 = create_convolutional_layer(input=layer_conv2,
                                             num_input_channels=num_filters_conv2,
                                             conv_filter_size=filter_size_conv3,
                                             num_filters=num_filters_conv3)

    layer_flat = create_flatten_layer(layer_conv3)

    layer_fc1 = create_fc_layer(input=layer_flat,
                                num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                                num_outputs=fc_layer_size,
                                use_relu=True)

    layer_fc2 = create_fc_layer(input=layer_fc1,
                                num_inputs=fc_layer_size,
                                num_outputs=num_classes,
                                use_relu=False)

    return layer_fc2


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss, session, accuracy):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))



def main():



    img_size = 128
    num_channels = 3
    classes = ['dog', 'cat']
    num_classes = len(classes)
    data = prepareTrainingData(validation_size=0.2, img_size=img_size, train_path="train", classes=classes)

    # convnet params
    filter_size_conv1 = 3
    num_filters_conv1 = 32

    filter_size_conv2 = 3
    num_filters_conv2 = 32

    filter_size_conv3 = 3
    num_filters_conv3 = 64

    fc_layer_size = 128

    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

        ## labels
        y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
        y_true_cls = tf.argmax(y_true, dimension=1)

        outputLayer = buildCnn(x,num_channels=num_channels,num_classes=num_classes,filter_size_conv1=filter_size_conv1,
                 num_filters_conv1=num_filters_conv1,filter_size_conv2=filter_size_conv2,num_filters_conv2=num_filters_conv2,
                 filter_size_conv3=filter_size_conv3,num_filters_conv3=num_filters_conv3,fc_layer_size=fc_layer_size)


        curGraph = tf.get_default_graph()
        for op in curGraph.get_operations():
            print("name: ", op.name)
            print("optype: ",op.type)
        y_pred = tf.nn.softmax(outputLayer,name="y_pred")
        y_pred_cls = tf.argmax(y_pred, dimension=1)

        crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=outputLayer,labels=y_true)
        cost = tf.reduce_mean(crossEntropy)

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

        correctPrediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        batchSize = 16
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for i in range(0, 1000):
            xBatch, yTrueBatch, _, clsBatch = data.train.next_batch(batchSize)
            xValidBatch, yValidBatch, _, validClsBatch = data.valid.next_batch(batchSize)
            print("training... step: "+str(i))
            feedDictTr = {x: xBatch, y_true: yValidBatch}
            feedDictVal = {x: xValidBatch, y_true: yValidBatch}
            sess.run(optimizer, feed_dict=feedDictTr)
            if i % int(data.train.num_examples / batchSize) == 0:
                val_loss = sess.run(cost, feed_dict=feedDictVal)
                epoch = int(i / int(data.train.num_examples / batchSize))
                show_progress(epoch, feedDictTr, feedDictVal, val_loss, sess, accuracy)
                saver.save(sess,'./dogs-cats-model/dogs-cats-model')

                #a = tf.truncated_normal([16, 128, 128, 3])
    #b = tf.reshape(a, [16,49152])

    #x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')


    #with tf.Session() as sess:
    #    sess.run(tf.global_variables_initializer())
    #    print sess.run(tf.shape(b))





if __name__ == "__main__":
    main()