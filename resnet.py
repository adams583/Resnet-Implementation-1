import numpy as np
import pickle
import tensorflow as tf 
import os, sys

### Process CIFAR-10 data files ###

# Unpickle pickled file formats and return as dictionary 
def unpickle(file):
    file = "cifar-10-data/"+file
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batches_meta = unpickle("batches.meta") # contains labels
db1 = unpickle("data_batch_1") # data batch 1
db2 = unpickle("data_batch_2") # data batch 2
db3 = unpickle("data_batch_3") # etc. 
db4 = unpickle("data_batch_4")
db5 = unpickle("data_batch_5")
test_batch = unpickle("test_batch") # testing batch

# Put all training data together and all testing data together 
X_train = np.concatenate((db1[b'data'], db2[b'data'], db3[b'data'], db4[b'data'], db5[b'data']))
y_train_initial = np.concatenate((db1[b'labels'], db2[b'labels'], db3[b'labels'], db4[b'labels'], db5[b'labels']))
X_test = test_batch[b'data']
y_test_initial = test_batch[b'labels']
train_fnames = np.concatenate((db1[b'filenames'], db2[b'filenames'], db3[b'filenames'], db4[b'filenames'], db5[b'filenames']))
test_fnames = test_batch[b'filenames']

y_train = np.zeros((50000,10))
y_train[np.arange(50000), y_train_initial] = 1
y_test = np.zeros((10000,10))
y_test[np.arange(10000), y_test_initial] = 1

# X_train contains data for 50,000 training images flattened into [1x3072] arrays,
# y_train contains their 50,000 labels.
# X_test contains data for 10,000 test images flattened into [1x3072] arrays,
# y_test contains their labels. 
# train_fnames contains the image filenames for the training set
# test_fnames contains the image filnames for the test set

"""Define functions to help build the network"""
### Define a function that will create a convolutional layer
def conv_layer(input, input_depth, num_filters, filter_size = [3,3], stride=[1,1,1,1]):
    """Creates a convolutional layer and returns its output"""
    
    # Initialize weights and biases
    weights_shape = [filter_size[0],filter_size[1],input_depth, num_filters]
    weights = tf.Variable(tf.truncated_normal(weights_shape,stddev=.05))
    biases = tf.Variable(tf.zeros([num_filters]))
    
    # Convolutional op 
    # Strides: use [1, stride_horizontal, stride_vertical, 1]
    # 'SAME' padding gives output 1/stride times the input size
    output = tf.nn.conv2d(input, weights, stride, padding='SAME') 
    output += biases # Added biases
    
    return output # Return output of convolutional layer


# Max pool layers will halve the size of the input with filter size and stride 2
def maxPool(input):
    return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Batch normalization 
def batchNorm(input):
    # Calculating mean and variance of input 
    mean, variance = tf.nn.moments(input, axes=[0]) 
    return tf.nn.batch_normalization(input, mean, variance, None, None, epsilon)

# ReLU
def ReLU(input):
    return tf.nn.relu(input)

# Average pool
def avgPool(input):
    return tf.nn.avg_pool(input, ksize=[1,2,2,1],strides=[1,1,1,1], padding='SAME')

def resUnit(input, input_depth, num_filters, filter_size = [3,3], stride=[1,1,1,1], downsample=False):
    """ Each residual unit will have two convolutional layers, each preceded by batch norm and ReLU. An identity 
    mapping of the input will be added on at the end (the residual part) 
    
    Stride should be [1,1,1,1] for any non-downsampling unit
    num_filters will be the new depth """

    # First step is batch normalization then ReLU
    reluBN = ReLU(batchNorm(input))
    if downsample: # If this unit will reduce dimensionality
        # Match dimensions of input to res unit output (depth might be different and width and height will be halved)
        input = conv_layer(input, input_depth, num_filters, [2,2], [1,2,2,1])
        conv1 = conv_layer(reluBN, input_depth, num_filters, filter_size, [1,2,2,1])
        conv1 = ReLU(batchNorm(conv1))
        conv2 = conv_layer(conv1, num_filters, num_filters, filter_size, stride) 
        return ReLU(batchNorm(conv2)) + input # add "identity mapping"
    
    else: 
        # Match dimensions of input to res unit output (depth might be different)
        input = conv_layer(input, input_depth, num_filters, [1,1], [1,1,1,1])
        conv1 = conv_layer(reluBN, input_depth, num_filters, filter_size, stride)
        conv1 = ReLU(batchNorm(conv1))
        conv2 = conv_layer(conv1, num_filters, num_filters, filter_size, stride)
        return ReLU(batchNorm(conv2)) + input # add "identity mapping" 


"""Begin setting up the network"""
### Set hyperparameters 
learning_rate = 0.0001 
epochs = 10
batch_size = 100 
# Small epsilon value for the batch normalization transform (this is needed in bn to avoid division by zero)
epsilon = 0.001

### Define placeholders that will be fed the data
# The images in CIFAR-10 are 32x32 with 3 color channels
X = tf.placeholder(tf.float32, shape=[None, 3072]) # [samples x 3072] for images that are 32 x 32 x 3 = 3072
y = tf.placeholder(tf.float32, shape=[None, 10]) # [samples x 10] for 10 classes of images
# We need to reshape X to be able to input it into a conv2d layer
X_reshaped = tf.reshape(X, [-1,32,32,3]) # -1 for variable number of samples

### Build planned architecture
firstConv = conv_layer(X_reshaped, 3, 64, [7,7]) # 7x7 convolutional layer
maxpool = maxPool(firstConv) # Max pool /2 
firstRes = resUnit(maxpool, 64, 64) # Filter size 3x3, no downsampling
secondRes = resUnit(firstRes, 64, 64) # Same as above 
thirdRes = resUnit(secondRes, 64, 128, downsample=True) # Double number of filters, downsample
fourthRes = resUnit(thirdRes, 128, 128)
fifthRes = resUnit(fourthRes, 128, 256, downsample=True) # Double filters, downsample again
sixthRes = resUnit(fifthRes, 256, 256)
seventhRes = resUnit(sixthRes, 256, 256)
avgpool = avgPool(seventhRes) # Average pool, preserve dimensions

### Dense layers 
# Input to first dense layer will be size 4 x 4 x 256 = 4096
# Because the input started at 32x32, downsampled 3 times, 32/(2^3) = 4 

# Initialize weights
dense_weights = {
    'W1': tf.Variable(tf.truncated_normal([4096,512],stddev=0.05)),
    'W2': tf.Variable(tf.truncated_normal([512,10],stddev=0.05))
}

# Initialize biases
dense_biases = {
    'b1': tf.Variable(tf.zeros(512)),
    'b2': tf.Variable(tf.zeros(10))
}

# Previous layer outputs a 4d shape, we need a 2d shape
avgpool_flattened = tf.reshape(avgpool, shape=[-1,4096])

dense1 = tf.nn.relu(tf.matmul(avgpool_flattened, dense_weights['W1']))
dense1 += dense_biases['b1']
dense2 = tf.matmul(dense1, dense_weights['W2'])
dense2 += dense_biases['b2']

# Softmax scores we get after data is fed through network--used for accuracy readings
scores = tf.nn.softmax(dense2)

### Define a loss function
# Cross entropy loss. Note we don't do this on scores because the function already includes softmax
loss = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=dense2)
loss = tf.reduce_mean(loss) # So we get one number rather than loss per sample

# Use an Adam optimizer to train network
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# For printing accuracy
true_preds = tf.equal(tf.argmax(y, 1), tf.argmax(scores, 1))
accuracy = tf.reduce_mean(tf.cast(true_preds, tf.float32))

# Initializer for global variables, to be run at the beginning of a session
init = tf.global_variables_initializer(); 


"""Run session to train and test model"""

# Initializer for global variables, to be run at the beginning of a session
init = tf.global_variables_initializer()

# Begin session
with tf.Session() as sess: 
    sess.run(init)
    
    # Structure:

    # for each epoch: 
    # . for each batch:
    # . . create batch
    # . . run backprop/optimizer by feeding in batch
    # . . find loss for the batch
    # . print loss for the epoch
    # . print testing accuracy for each epoch
    
    num_samples = X_train.shape[0] # Number of samples
    num_batches = np.int32(num_samples/batch_size) # Number of batches
    indices = np.arange(num_samples) # The indices to be used in batch creation
    for e in range(epochs):
        # Get new batch divides for each epoch
        np.random.shuffle(indices)
        # Initialize loss at the start of each epoch, to be added onto iteratively
        epoch_loss = 0
        for i in range(num_batches):
            # Create new batch for each batch iteration
            batch_indices = indices[i*batch_size:(i+1)*batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            # Run optimizer by feeding in batch, find batch_loss
            _, batch_loss = sess.run([optimizer, loss], feed_dict={X:X_batch, y:y_batch})
            # Accumulate loss
            epoch_loss += batch_loss/num_samples
            if i%25 == 0:
                # Progress updates to check that training is progressing
                print(i/num_batches, "complete with this epoch") 
        # Get accuracy for test data
        acc = sess.run(accuracy, feed_dict={X: X_test, y: y_test})
        print(f"Epoch {e+1}: loss = {epoch_loss}, accuracy = {acc}")
        
sess.close() # Done