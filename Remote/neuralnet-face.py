"""Train the images to create model that detects if the eyes are closed"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

import random
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def extract(img_file):
    gray_img = cv2.imread(img_file, flags=0)  # grayscale
    #print(gray_img.shape)
    gray_img = cv2.resize(gray_img, (40, 80))
    #print('Gray shape:', gray_img.shape)
    #Turning it into gray-scale

    #plt.imshow(gray_img)
    #plt.imshow(gray_img, plt.cm.binary)
    #plt.show()

    """
    print(gray_img)
    """
    #Turning 2D into 1D
    pre_data = np.asarray(gray_img)
    pre_data = pre_data.flatten()
    """
    print(pre_data)
    """
    return pre_data



def display_data(array_mat):

    """reshaping the array back into 20x40 pixels for display"""
    img = array_mat.reshape(80,40)
    """#img = np.transpose(img) #arranging the pixel intensities(array)"""
    
    imgplot = plt.imshow(img, cmap=plt.cm.binary)
    plt.show()
    #plt.draw()
    #plt.pause(0.001)



def neural_network_model(data):
    #(input_data * weights) + biases
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([3200, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    """
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    """
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}


    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    #rectified linear: relu
    #l1 = tf.nn.relu(l1)
    #l1 = tf.nn.softmax(l1)
    l1 = tf.nn.sigmoid(l1)
                        
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    #rectified linear: relu
    #l2 = tf.nn.relu(l2)
    l2 = tf.nn.sigmoid(l2)

    output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']

    return output




def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)


    init_op = tf.global_variables_initializer()

    """Initialization for saving purpose"""
    saver = tf.train.Saver()


    
    with tf.Session() as sess:
        #old-version:  sess.run(tf.initialize_all_variables())
        #new-version:
        sess.run(init_op)

        #print(train_x.shape)

        for epoch in range(4000):
            #train_x = np.reshape(train_x, (-1, 800))
                        
                        
            _, c = sess.run([train_step, cost], feed_dict = {x: train_x, y: train_y})
            
            print('Epoch', epoch, 'completed out of 100.  loss:', c)


        """Saving the model"""
        save_path = saver.save(sess, model_path)    
        print("Model saved in file: %s" %save_path)
        
        
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))    
        print("Accuracy: ", (sess.run(accuracy, feed_dict={x: test_x, y: test_y})) )



        print("Displaying Test Data:")
        #plt.ion()
        #plt.show()
        for i in range(m_of_X):
            arr = random.randint(1, 75)
            print("random: ", arr)
            display_data(test_x[arr])
            
            #print("Prediction: ", (sess.run(tf.argmax(prediction,1), feed_dict={x: [X[arr]] }) ) )
            output = (sess.run(tf.argmax(prediction,1), feed_dict={x: [test_x[arr]] }) )
            print("Prediction: ")
            print(output)

            print
            print
            print
            print

            if output==0:
                print(" " + " " + " " + " " + "Eye closed")
            else:
                print(" " + " " + " " + " " + " Eye is open")

            print("Accuracy: ", (sess.run(accuracy, feed_dict={x: [test_x[arr]], y: [test_y[arr]]})) )

            print
            print
            print
            

            s = input("Paused; Enter 'e' & 'enter' to exit, 'enter' to continue: ")
            if s == 'e':
                break


model_path = "C:\\Users\\Dell\\Desktop\\Projects\\majorproj\\faceMatching\\model_trained\\modelusereye5\\modeluser-eye-5"




"""<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""

img_file = 'C:/Users/Dell/Desktop/Projects/majorproj/faceMatching/close/close_9.png'

data_x = np.empty(shape=(1,3200))
data_x[0] = extract(img_file)

data_y = [[1,0]]

print("first batch")
for i in range(1,577):
    img_file = 'C:/Users/Dell/Desktop/Projects/majorproj/faceMatching/close/close_' + str(i) + '.png'

    xx = extract(img_file)
    data_x = np.vstack((data_x, xx))

    yy = [1,0]
    data_y = np.vstack((data_y, yy))

            


print("second batch")

for i in range(1,244):
    img_file = 'C:/Users/Dell/Desktop/Projects/majorproj/faceMatching/open/open_' + str(i) + '.png'

    xx = extract(img_file)
    data_x = np.vstack((data_x, xx))

    yy = [0,1]
    data_y = np.vstack((data_y, yy))

    
    

print("Training starts")
#print(data_x)
#print(data_y)

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

X = data_x
Y = data_y


m_of_X, f_of_X = X.shape

n_nodes_hl1 = 280   #46
n_nodes_hl2 = 80
#n_nodes_hl3 = 500

#no. of labels
n_classes = 2    #2


#20x40 pixels image = flat 800 pixels(features of each example in data)
x = tf.placeholder('float', [None, 3200])
y = tf.placeholder('float')

X, Y = shuffle(X, Y, random_state=1)
X, Y = shuffle(X, Y, random_state=1)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20)

train_neural_network(x)


print("finidh")



