
# coding: utf-8

# In[1]

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import cPickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#get_ipython().magic(u'matplotlib inline')

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


# In[2]:

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict




train_dataset = np.ndarray(shape =(50000,32,32,3))
train_labels_data = np.ndarray(shape=(50000))
j = 0
for i in range(1,6):
    p = unpickle("cifar-10-batches-py/data_batch_" + str(i))
    #print(p['data'].shape)
    
    
    train_dataset[j:j+10000:,:,:]= np.array(p['data']).reshape((10000,3,32,32)).transpose(0,2,3,1)
    #train_dataset[j:j+10000] = np.array(p['data'])
    train_labels_data[j:j+10000]= np.array(p['labels'])
    j += 10000
p = unpickle("cifar-10-batches-py/test_batch")



train_labels=np.zeros((50000,10),dtype=np.float32)
for i in range(50000):
    a=train_labels_data[i]
    train_labels[i,a]=1.0

#img_R = p['data'][0][0:1024].reshape((32, 32))
#img_G = p['data'][0][1024:2048].reshape((32, 32))
#img_B = p['data'][0][2048:3072].reshape((32, 32))
#img = np.dstack((img_R, img_G, img_B))

#imgplot = plt.imshow( img )
#plt.show()
#plt.imshow(train_dataset[0])
#plt.show()
test_dataset = np.array(p['data']).reshape((len(p['data']),3,32,32)).transpose(0,2,3,1)
test_labels_data = np.array(p['labels'])


test_labels=np.zeros((len(p['data']),10),dtype=np.float32)
for i in range(len(p['data'])):
    a=test_labels_data[i]
    test_labels[i,a]=1.0


#train_dataset = np.array(train_dataset).reshape([len(train_dataset,32,3])
#train_labels = np.array(train_labels).flatten()





def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset,train_labels)
test_dataset, test_labels = randomize(test_dataset,test_labels)

#make valid test data sets
valid_dataset = train_dataset[:5000]
valid_labels = train_labels[:5000]
train_dataset = train_dataset[5000:]
train_labels = train_labels[5000:]
#valid_dataset = train_dataset[: ::]


meta = unpickle('cifar-10-batches-py/batches.meta')
print (meta)
label_names = meta['label_names']
#batch_size = meta['num_cases_per_batch']


# In[3]:

X = tf.placeholder(tf.float32, shape=(32, 32, 3))
#conv1 = 
print(X.get_shape)


# In[4]:

num_labels = len(label_names)


#1-hot encoding
'''
def reformat(labels):
  #dataset = dataset.reshape(
  #  (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return  labels
'''
#train_labels = reformat(train_labels)
#valid_labels = reformat(valid_labels)
#test_labels = reformat(test_labels)


print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)





# In[5]:

'''
image_size = 32
num_channels = 3 
#num_labels = len(label_names)

filter1 = 4
layer1_stride = 1
depth1 = 48
filter2 = 3
layer2_stride = 2
depth2 = 96
filter3 = 2
layer3_stride = 1
depth3 = 256
filter4 = 2
layer4_stride = 1
depth4 = 256

layer1_weights = tf.Variable(tf.truncated_normal(
  [filter1, filter1, num_channels, depth1], stddev=0.1))
layer1_biases = tf.Variable(tf.zeros([depth1]))

layer2_weights = tf.Variable(tf.truncated_normal(
  [filter2, filter2, depth1, depth2], stddev=0.1))
layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth2]))


layer3_weights = tf.Variable(tf.truncated_normal(
    [filter3,filter3,depth2,depth3],stddev=0.1))
layer3_biases = tf.Variable(tf.constant(1.0,shape=[depth3]))


layer4_weights = tf.Variable(tf.truncated_normal(
    [filter4,filter4,depth3,depth4],stddev = 0.1))
layer4_biases = tf.Variable(tf.constant(1.0,shape=[depth4]))

layer5_weights = tf.Variable(tf.truncated_normal(
    [4096,num_hidden],stddev = 0.1))
layer5_biases = tf.Variable(tf.constant(1.0,shape=[num_hidden]))

layer6_weights = tf.Variable(tf.truncated_normal(
    [num_hidden,num_labels],stddev = 0.1))
layer6_biases = tf.Variable(tf.constant(1.0,shape=[num_labels]))

data = tf.placeholder(tf.float32, shape=(128, 32, 32, 3))
conv = tf.nn.conv2d(data, layer1_weights, [1, layer1_stride, layer1_stride, 1], padding='SAME')
print(conv.get_shape())
hidden = tf.nn.relu(conv + layer1_biases)
hidden = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding = 'SAME')
print(hidden.get_shape())

#layer2
conv = tf.nn.conv2d(hidden, layer2_weights, [1, layer2_stride, layer2_stride, 1], padding='SAME')
print(conv.get_shape())
hidden = tf.nn.relu(conv + layer2_biases)
hidden = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding = 'SAME')
print(hidden.get_shape())

#layer3
conv = tf.nn.conv2d(hidden, layer3_weights, [1, layer3_stride, layer3_stride, 1], padding='SAME')
print(conv.get_shape())
hidden = tf.nn.relu(conv + layer3_biases)
print(hidden.get_shape())
#hidden = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding = 'SAME')

#layer4                       
conv = tf.nn.conv2d(hidden, layer4_weights, [1, layer4_stride, layer4_stride, 1], padding='SAME')
hidden = tf.nn.relu(conv + layer4_biases)

shape = hidden.get_shape().as_list()
reshape = tf.reshape(hidden,[shape[0],shape[1]*shape[2]*shape[3]])
hidden = tf.nn.relu(tf.matmul(reshape,layer5_weights) + layer5_biases)

print(hidden.get_shape())
hidden = tf.matmul(hidden,layer6_weights) + layer6_biases

print(hidden.get_shape())
'''


# In[6]:

'''
image_size = 32
num_channels = 3 
#num_labels = len(label_names)

filter1 = 3
layer1_stride = 1
depth1 = 64
filter2 = 3
layer2_stride = 1
depth2 = 128
filter3 = 3
layer3_stride = 1
depth3 = 256
filter4 = 3
layer4_stride = 1
depth4 = 512



batch_size = 32
num_hidden = 2048
graph = tf.Graph()

tf.reset_default_graph()
with graph.as_default():
    
    # Input data.
    tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_valid_labels = tf.constant(valid_labels)
    tf_test_dataset = tf.constant(test_dataset)
    tf_test_labels = tf.constant(test_labels)
    
    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
      [filter1, filter1, num_channels, depth1], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth1]))
    
    layer2_weights = tf.Variable(tf.truncated_normal(
      [filter2, filter2, depth1, depth1], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(0.0, shape=[depth1]))
    
    
    layer3_weights = tf.Variable(tf.truncated_normal(
        [filter3,filter3,depth1,depth2],stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(0.0,shape=[depth2]))
    
    
    layer4_weights = tf.Variable(tf.truncated_normal(
        [filter4,filter4,depth2,depth2],stddev = 0.1))
    layer4_biases = tf.Variable(tf.constant(0.0,shape=[depth2]))
    
    
    layer5_weights = tf.Variable(tf.truncated_normal(
      [filter1, filter1, depth2, depth3], stddev=0.1))
    layer5_biases = tf.Variable(tf.zeros([depth3]))
    
    layer6_weights = tf.Variable(tf.truncated_normal(
      [filter1, filter1, depth3, depth3], stddev=0.1))
    layer6_biases = tf.Variable(tf.zeros([depth3]))
    
    layer7_weights = tf.Variable(tf.truncated_normal(
      [filter1, filter1, depth3, depth4], stddev=0.1))
    layer7_biases = tf.Variable(tf.zeros([depth4]))
    
    
    
    
    
    layer8_weights = tf.Variable(tf.truncated_normal(
      [filter1, filter1, depth4, depth4], stddev=0.1))
    layer8_biases = tf.Variable(tf.zeros([depth4]))
    
    
    layer9_weights = tf.Variable(tf.truncated_normal(
      [filter1, filter1, depth4, depth4], stddev=0.1))
    layer9_biases = tf.Variable(tf.zeros([depth4]))
    
    
    layer10_weights = tf.Variable(tf.truncated_normal(
      [filter1, filter1, depth4, depth4], stddev=0.1))
    layer10_biases = tf.Variable(tf.zeros([depth4]))
    
    
    
    
    layer11_weights = tf.Variable(tf.truncated_normal(
        [8192,num_hidden],stddev = 0.1))
    layer11_biases = tf.Variable(tf.constant(0.0,shape=[num_hidden]))
    
    layer12_weights = tf.Variable(tf.truncated_normal(
        [num_hidden,num_hidden],stddev = 0.1))
    layer12_biases = tf.Variable(tf.constant(0.0,shape=[num_hidden]))
    
    layer13_weights = tf.Variable(tf.truncated_normal(
        [num_hidden,num_labels],stddev = 0.1))
    layer13_biases = tf.Variable(tf.constant(0.0,shape=[num_labels]))
    
  # Model.
    def model(data,train = False):
        
        data = tf.to_float(data)
        
        #layer1
        conv = tf.nn.conv2d(data, layer1_weights, [1, layer1_stride, layer1_stride, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        if(train):
            hidden = tf.nn.dropout(hidden,0.5)
        #layer2
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        #hidden = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding = 'SAME')
        if(train):
            hidden = tf.nn.dropout(hidden,0.5)
        
        
        
        #layer3
        conv = tf.nn.conv2d(hidden, layer3_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer3_biases) 
        if(train):
            hidden = tf.nn.dropout(hidden,0.5)
        
        #layer4                       
        conv = tf.nn.conv2d(hidden, layer4_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer4_biases)
        hidden = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding = 'SAME')
        if(train):
            hidden = tf.nn.dropout(hidden,0.5)
        
        
        #layer5
        conv = tf.nn.conv2d(hidden, layer5_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer5_biases)
        if(train):
            hidden = tf.nn.dropout(hidden,0.5)
        
        #layer6                       
        conv = tf.nn.conv2d(hidden, layer6_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer6_biases)
        #hidden = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding = 'SAME')
        #layer7
        conv = tf.nn.conv2d(hidden, layer7_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer7_biases)
        hidden = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding = 'SAME')
        if(train):
            hidden = tf.nn.dropout(hidden,0.5)
        
        

        #layer8                      
        conv = tf.nn.conv2d(hidden, layer8_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer8_biases)
        if(train):
            hidden = tf.nn.dropout(hidden,0.5)
        
        #layer9
        conv = tf.nn.conv2d(hidden, layer9_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer9_biases)
        if(train):
            hidden = tf.nn.dropout(hidden,0.5)
        
        #layer10                     
        conv = tf.nn.conv2d(hidden, layer10_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer10_biases)
        if(train):
            hidden = tf.nn.dropout(hidden,0.5)
        
        hidden = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding = 'SAME')

        
        
        
        
        
        
      
        shape = hidden.get_shape().as_list()
        print(shape)
        reshape = tf.reshape(hidden,[shape[0],shape[1]*shape[2]*shape[3]])
        
        hidden = tf.nn.relu(tf.matmul(reshape,layer11_weights) + layer11_biases)
        if train:
            hidden = tf.nn.dropout(hidden,0.5)
        #hidden = tf.nn.relu(tf.matmul(hidden,layer12_weights)) + layer12_biases
        
        if train:
            hidden = tf.nn.dropout(hidden,0.75)
        hidden = tf.matmul(hidden,layer13_weights) + layer13_biases
        return hidden
 
       
        # Training computation.
    logits = model(tf_train_dataset,True)
    loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
    #optimizer = tf.train.MomentumOptimizer(0.0001,0.6).minimize(loss)
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))
'''


# In[8]:

image_size = 32
num_channels = 3 
#num_labels = len(label_names)

filter1 = 3
layer1_stride = 1
depth1 = 32
filter2 = 3
layer2_stride = 1
depth2 = 64
filter3 = 3
layer3_stride = 1
depth3 = 128
filter4 = 3
layer4_stride = 1
depth4 = 256

drop_prob = 0.75

batch_size = 30
num_hidden = 1024
graph = tf.Graph()

BN_EPSILON = 0.1

tf.reset_default_graph()
with graph.as_default():
    
    # Input data.
    tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_valid_labels = tf.constant(valid_labels)
    tf_test_dataset = tf.constant(test_dataset)
    tf_test_labels = tf.constant(test_labels)
    
    
    
    layer0_ab = tf.Variable(tf.constant(1.0,shape = [2,num_channels]))
    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
      [filter1, filter1, num_channels, depth1], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth1]))
    layer1_ab = tf.Variable(tf.constant(1.0,shape = [2,depth1]))
    
    
    
    layer2_weights = tf.Variable(tf.truncated_normal(
      [filter2, filter2, depth1, depth1], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(0.0, shape=[depth1]))
    layer2_ab = tf.Variable(tf.constant(1.0,shape = [2,depth1]))
    
    layer3_weights = tf.Variable(tf.truncated_normal(
        [filter3,filter3,depth1,depth2],stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(0.0,shape=[depth2]))
    layer3_ab = tf.Variable(tf.constant(1.0,shape = [2,depth2]))
    
    layer4_weights = tf.Variable(tf.truncated_normal(
        [filter4,filter4,depth2,depth2],stddev = 0.1))
    layer4_biases = tf.Variable(tf.constant(0.0,shape=[depth2]))
    layer4_ab = tf.Variable(tf.constant(1.0,shape = [2,depth2]))
    
    layer5_weights = tf.Variable(tf.truncated_normal(
      [filter1, filter1, depth2, depth3], stddev=0.1))
    layer5_biases = tf.Variable(tf.zeros([depth3]))
    layer5_ab = tf.Variable(tf.constant(1.0,shape = [2,depth3]))
    
    
    layer6_weights = tf.Variable(tf.truncated_normal(
      [filter1, filter1, depth3, depth3], stddev=0.1))
    layer6_biases = tf.Variable(tf.zeros([depth3]))
    layer6_ab = tf.Variable(tf.constant(1.0,shape = [2,depth3]))
    
    
    layer7_weights = tf.Variable(tf.truncated_normal(
      [filter1, filter1, depth3, depth4], stddev=0.1))
    layer7_biases = tf.Variable(tf.zeros([depth4]))
    layer7_ab = tf.Variable(tf.constant(1.0,shape = [2,depth4]))
    
    
    
    
    layer8_weights = tf.Variable(tf.truncated_normal(
      [filter1, filter1, depth4, depth4], stddev=0.1))
    layer8_biases = tf.Variable(tf.zeros([depth4]))
    layer8_ab = tf.Variable(tf.constant(1.0,shape = [2,depth4]))
    
    layer9_weights = tf.Variable(tf.truncated_normal(
      [filter1, filter1, depth4, depth4], stddev=0.1))
    layer9_biases = tf.Variable(tf.zeros([depth4]))
    layer9_ab = tf.Variable(tf.constant(1.0,shape = [2,depth4]))
    
    layer10_weights = tf.Variable(tf.truncated_normal(
      [filter1, filter1, depth4, depth4], stddev=0.1))
    layer10_biases = tf.Variable(tf.zeros([depth4]))
    layer10_ab = tf.Variable(tf.constant(1.0,shape = [2,depth4]))
    
    
    
    layer11_weights = tf.Variable(tf.truncated_normal(
        [4096,num_hidden],stddev = 0.1))
    layer11_biases = tf.Variable(tf.constant(0.0,shape=[num_hidden]))
    layer11_ab = tf.Variable(tf.constant(1.0,shape = [2]))
    
    layer12_weights = tf.Variable(tf.truncated_normal(
        [num_hidden,num_hidden],stddev = 0.1))
    layer12_biases = tf.Variable(tf.constant(0.0,shape=[num_hidden]))
    layer12_ab = tf.Variable(tf.constant(1.0,shape = [2]))
    
    layer13_weights = tf.Variable(tf.truncated_normal(
        [num_hidden,num_labels],stddev = 0.1))
    layer13_biases = tf.Variable(tf.constant(1.0,shape=[num_labels]))
    layer13_ab = tf.Variable(tf.constant(1.0,shape = [2]))
    
    
    
        
    
    
  # Model.
    def model(data,train = False):
        
        data = tf.to_float(data)
        axis = range(len(data.get_shape())  - 1)
        mean, variance = tf.nn.moments(data,axis)
        data = tf.nn.batch_normalization(data,mean,variance,layer0_ab[0],layer0_ab[1],BN_EPSILON)
        #layer1
        conv = tf.nn.conv2d(data, layer1_weights, [1, layer1_stride, layer1_stride, 1], padding='SAME')
        axis = range(len(conv.get_shape())  - 1)
        mean, variance = tf.nn.moments(conv,axis)
        conv = tf.nn.batch_normalization(conv,mean,variance,layer1_ab[0],layer1_ab[1],BN_EPSILON)
        
        hidden = tf.nn.relu(conv)        
        
        
        #if(train):
        #    hidden = tf.nn.dropout(hidden,drop_prob)
        #layer2
        hidden = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
        axis = range(len(hidden.get_shape())  - 1)
        mean, variance = tf.nn.moments(hidden,axis)
        hidden = tf.nn.batch_normalization(hidden,mean,variance,layer2_ab[0],layer2_ab[1],BN_EPSILON)
        
        hidden = tf.nn.relu(hidden)
        
        #hidden = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding = 'SAME')
        #if(train):
        #    hidden = tf.nn.dropout(hidden,drop_prob)
        
        
        
        #layer3
        hidden = tf.nn.conv2d(hidden, layer3_weights, [1, 1, 1, 1], padding='SAME')
        
        axis = range(len(hidden.get_shape())  - 1)
        mean, variance = tf.nn.moments(hidden,axis)
        hidden = tf.nn.batch_normalization(hidden,mean,variance,layer3_ab[0],layer3_ab[1],BN_EPSILON)
        
        hidden = tf.nn.relu(hidden) 
        
        
        #if(train):
        #    hidden = tf.nn.dropout(hidden,drop_prob)
        
        #layer4                       
        hidden = tf.nn.conv2d(hidden, layer4_weights, [1, 1, 1, 1], padding='SAME')
        axis = range(len(hidden.get_shape())  - 1)
        mean, variance = tf.nn.moments(hidden,axis)
        hidden = tf.nn.batch_normalization(hidden,mean,variance,layer4_ab[0],layer4_ab[1],BN_EPSILON)
        
        hidden = tf.nn.relu(hidden) 
        hidden = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding = 'SAME')
        #if(train):
        #    hidden = tf.nn.dropout(hidden,drop_prob)
        
        
        #layer5
        hidden = tf.nn.conv2d(hidden, layer5_weights, [1, 1, 1, 1], padding='SAME')
        
        axis = range(len(hidden.get_shape())  - 1)
        mean, variance = tf.nn.moments(hidden,axis)
        hidden = tf.nn.batch_normalization(hidden,mean,variance,layer5_ab[0],layer5_ab[1],BN_EPSILON)
        
        hidden = tf.nn.relu(hidden)
        hidden = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding = 'SAME')
        
        
        #if(train):
        #    hidden = tf.nn.dropout(hidden,drop_prob)
        
        #layer6                       
        #conv = tf.nn.conv2d(hidden, layer6_weights, [1, 1, 1, 1], padding='SAME')
        #hidden = tf.nn.relu(conv + layer6_biases)
        #hidden = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding = 'SAME')
        #layer7
        hidden = tf.nn.conv2d(hidden, layer7_weights, [1, 1, 1, 1], padding='SAME')
        axis = range(len(hidden.get_shape())  - 1)
        mean, variance = tf.nn.moments(hidden,axis)
        hidden = tf.nn.batch_normalization(hidden,mean,variance,layer7_ab[0],layer7_ab[1],BN_EPSILON)
        
        hidden = tf.nn.relu(hidden)
        hidden = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding = 'SAME')
        #if(train):
        #    hidden = tf.nn.dropout(hidden,drop_prob)
        
        

        #layer8                      
        hidden = tf.nn.conv2d(hidden, layer8_weights, [1, 1, 1, 1], padding='SAME')
        axis = range(len(hidden.get_shape())  - 1)
        mean, variance = tf.nn.moments(hidden,axis)
        hidden = tf.nn.batch_normalization(hidden,mean,variance,layer8_ab[0],layer8_ab[1],BN_EPSILON)
        
        
        hidden = tf.nn.relu(hidden)
        #if(train):
        #    hidden = tf.nn.dropout(hidden,drop_prob)
        
        #layer9
        #conv = tf.nn.conv2d(hidden, layer9_weights, [1, 1, 1, 1], padding='SAME')
        #hidden = tf.nn.relu(conv + layer9_biases)
        #if(train):
        #    hidden = tf.nn.dropout(hidden,drop_prob)
        
        #layer10                     
        #conv = tf.nn.conv2d(hidden, layer10_weights, [1, 1, 1, 1], padding='SAME')
        #hidden = tf.nn.relu(conv + layer10_biases)
        #if(train):
        #    hidden = tf.nn.dropout(hidden,drop_prob)
        
        #hidden = tf.nn.max_pool(hidden,[1,2,2,1],[1,2,2,1],padding = 'SAME')

        
        
        
        
        
        
      
        shape = hidden.get_shape().as_list()
        print(shape)
        reshape = tf.reshape(hidden,[shape[0],shape[1]*shape[2]*shape[3]])
        hidden = tf.matmul(reshape,layer11_weights)
        mean, variance = tf.nn.moments(hidden,[0])
        hidden = tf.nn.batch_normalization(hidden,mean,variance,layer11_ab[0],layer11_ab[1],BN_EPSILON)
        
        hidden = tf.nn.relu(hidden)
        
        
        #if train:
        #    hidden = tf.nn.dropout(hidden,drop_prob)
        #hidden = tf.nn.relu(tf.matmul(hidden,layer12_weights)) + layer12_biases
        
        #if train:
        #    hidden = tf.nn.dropout(hidden,0.75)
        
        hidden = tf.matmul(hidden,layer13_weights)
        mean, variance = tf.nn.moments(hidden,[0])
        hidden = tf.nn.batch_normalization(hidden,mean,variance,layer13_ab[0],layer13_ab[1],BN_EPSILON)
        
        hidden = tf.nn.relu(hidden)
        
        return hidden
 
       
        # Training computation.
    logits = model(tf_train_dataset,True)
    loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    #optimizer = tf.train.MomentumOptimizer(0.0001,0.9).minimize(loss)
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))


# In[ ]:

num_steps = 1500
dropout = 0.75
with tf.Session(graph=graph) as session:

    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

        batch_data = train_dataset[offset:(offset + batch_size)]
        batch_labels = train_labels[offset:(offset + batch_size)]


        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(
            valid_prediction.eval(), valid_labels))
    
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


# 
