"""Train the database to create a model for system"""

import random
import numpy as np
import sqlite3 as sq
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split




"""Path for saving the trained model"""
model_path = "C:\\Users\\Dell\\Desktop\\Projects\\majorproj\\model\\object\\reserve"
#model_path = "/home/pi/Desktop/work/bankreserve/bank"






print("program started")

"""Database Extraction"""
conn = sq.connect('DB\\raspContainertest2.db')
c = conn.cursor()

#'out'  and   'qmax'   table
c.execute('SELECT * from qmax')

alist = c.fetchall()
X = np.array(alist)
#There is round off to 6 in the values received from sqlite.

c.execute('SELECT * from out')

alist = c.fetchall()
Y = np.array(alist)


c.execute('SELECT * from qmax1')

alist = c.fetchall()
X1 = np.array(alist)


c.close()
conn.close()
"""Database Closure"""




le = X.transpose()
ke = X1.transpose()
we = np.concatenate((le,ke))
X = we.transpose()



X[X>=165] = 230   #136
X[X<165] = 30




print("Data extracted..")




m_of_X, f_of_X = X.shape

n_nodes_hl1 = 1600 #1200 #280  #260   #46  #1600
#n_nodes_hl2 = #80 #50
#n_nodes_hl3 = 500

"""no. of labels"""
n_classes = 4    #2


"""20x40 pixels image = flat 800 pixels(features of each example in data)"""
x = tf.placeholder('float', [None, 3200])  #800
y = tf.placeholder('float')



X, Y = shuffle(X, Y, random_state=1)
X, Y = shuffle(X, Y, random_state=1)


train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20)



print(test_x[25])

def display_data(array_mat):

    """reshaping the array back into 20x40 pixels for display"""
    img = array_mat.reshape(40,80)
    #img = np.transpose(img) #arranging the pixel intensities(array)
    
    imgplot = plt.imshow(img, cmap=plt.cm.binary)
    plt.show()
    #plt.draw()
    #plt.pause(0.001)




def neural_network_model(data):
    #(input_data * weights) + biases
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([3200, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    """
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    """
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}


    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    #rectified linear: relu
    #l1 = tf.nn.relu(l1)
    #l1 = tf.nn.softmax(l1)
    #l1 = tf.nn.sigmoid(l1)
    l1 = tf.nn.tanh(l1)


    """                    
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.sigmoid(l2)
    """

    output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']

    return output





def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))

    #train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cost)
    #train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
    train_step = tf.train.AdamOptimizer().minimize(cost)

    


    init_op = tf.global_variables_initializer()

    """Initialization for saving purpose"""
    saver = tf.train.Saver()

 
    
    with tf.Session() as sess:
        #old-version:  sess.run(tf.initialize_all_variables())
        #new-version:
        sess.run(init_op)

        #print(train_x.shape)

        for epoch in range(300):
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
            arr = random.randint(1, 1000)
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

            if output==2:
                print(" " + " " + " " + " " + "right")
            elif output==1:
                print(" " + " " + " " + " " + "forward")
            elif output==0:
                print(" " + " " + " " + " " + "left")
            else:
                print(" " + " " + " " + " " + "Its sth else...")
                

            print("Accuracy: ", (sess.run(accuracy, feed_dict={x: [test_x[arr]], y: [test_y[arr]]})) )

            print
            print
            print
            

            s = input("Paused; Enter 'e' & 'enter' to exit, 'enter' to continue: ")
            if s == 'e':
                break


        
train_neural_network(x)


