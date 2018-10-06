"""Program to run on Raspberrypi Car"""

import controls as cn
import capture as cp
import connremote as so

import threading
import datetime
from time import sleep, time
from Queue import Queue

from mpu6050 import mpu6050

import sqlite3
import serial
import numpy as np

import tensorflow as tf

#import random
#from subprocess import call

#import matplotlib.pyplot as plt
#from sklearn.utils import shuffle
#from sklearn.model_selection import train_test_split



ser = serial.Serial('/dev/ttyACM0', 9600)
sensor = mpu6050(0x68)



key = 0
data = "q"
#For sending purpose
so.send_toRemote("hi")


"""<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
"""tf.reset_default_graph()"""


"""Defining model"""
def neural_network_model(data):
    #(input_data * weights) + biases
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([3200, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    """hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
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
                        
    """ l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    #rectified linear: relu
    #l2 = tf.nn.relu(l2)
    """

    output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']

    return output


"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""




def selfDrive(guide):
    """Tesla Car"""

    if guide == [1]:  #00
        cn.forward()
        #stop()
    elif guide == [2]:  #01
        cn.right()
        #stop()
    elif guide == [0]:
        cn.left()
        #stop()
    elif guide == [3]:
        cn.stop()


"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""


"""
prediction = neural_network_model(x)                                                                
                
init = tf.global_variables_initializer()
"""



def collect_data(q):
    sa = 0
    
    try:
        sa = q.get(False)
        #print(sa)
    except Exception:
        #print("no data")
        qwert = 1
        
    return sa    
        


def sensor(q):
    """Thread - 1  """
    print("Thread 1")
    sensor = mpu6050(0x68)


    comm = sqlite3.connect('/var/www/html/DBphp.db')
    cursor = comm.cursor()


    global notice
    notice = 0
    
    
    while 1:
        try:
            ans = ser.readline()
            b = ans[0]
            #print b
        except Exception:
            print "Error in arduino"


        accel_data = sensor.get_accel_data()
        gyro_data = sensor.get_gyro_data()
        temp = sensor.get_temp()

        
        """if change in accelo and gyro, temp found entry in database"""
        

        gy_x = gyro_data['x']
        gy_y = gyro_data['y']
        gy_z = gyro_data['z']

        accel_x = accel_data['x']
        accel_y = accel_data['y']
        accel_z = accel_data['z']
            
        temperature = temp

        if b == '4':
            #print "control ultra"
            notice = 4
            sleep(1) #with this sleep, the other thread will be able to read 'notice'
            notice = 0

            mq_sens = "Not detected"

        elif b == '8':
            so.send_toRemote("Alert: Smoke")
            mq_sens = "Detected"
           
        else:
            mq_sens = "Not detected"

            
        unix = time()
        date = str(datetime.datetime.fromtimestamp(unix).strftime('%Y-%m-%d %H:%M:%S'))
            
        cursor.execute("UPDATE qmax SET datestamp = ?, gyro_x = ?, gyro_y = ?, gyro_z = ?, acc_x = ?, acc_y = ?, temp = ?, mq2 = ? WHERE acc_z = 2", (date, gy_x, gy_y, gy_z, accel_x, accel_y, temperature, mq_sens))
        comm.commit()
        #date and mq_sens  are TEXT
        

        if collect_data(q) == 23:
            print("Terminating Thread 1")
            break




def modelCar(q):
    """Thread- 2  """
    print("thread 2")
    data = "xc"
    key = 0

    model_path = "/home/pi/Desktop/work/DB/reserve"


    g_1 = tf.Graph()
    with g_1.as_default():
        n_nodes_hl1 = 1600
        n_classes = 4

        x = tf.placeholder('float', [None, 3200])
        y = tf.placeholder('float')

        prediction = neural_network_model(x)
        saver = tf.train.Saver()


    sess = tf.Session(graph = g_1)
    #sess.run(init)
    saver.restore(sess, model_path)


    """a = 0"""
    msg = "hi"
        

    while True:
        print("namaste")

        try:
            data_1 = so.receive_fromRemote1()
                            
            if data_1 == "close":
                cn.stop()
                print("remote not working")
                
                while True:
                    so.receive_fromRemote1()

                    try:
                        so.receive_fromRemote()
                        data = "ef"
                            
                    except Exception:
                        asdf = 1
                        break   """"""
                    
                data = "ef"

        except Exception:
            try:
                data = so.receive_fromRemote()
                key = 1
                    
                if data == "man":
                    while True:
                            
                        X = cp.captureImage()

                        guide = (sess.run(tf.argmax(prediction,1), feed_dict={x: [X] } ))

                        selfDrive(guide)                                              
                        print("selfdrive")
                        
                        sleep(0.01)
                            

                        try:
                            data = so.receive_fromRemote()
                            key = 1
                            
                            if data == "man":
                                msg = "Manual_mode"
                                so.send_toRemote(msg)
                                key = 0
                                print("going to semi-manual")
                                break

                        except Exception:
                            print("nnn")
                            
                
            except Exception:
                print("der")



        if data == "ab":              """Forward"""
            X = cp.captureImage()
            predict = (sess.run(tf.argmax(prediction,1), feed_dict={x: [X] } ))

            if predict == [0]:
                cn.forward()
                 
            else:
                selfDrive(predict)

            print("forward")
            
            if key == 1:
                so.send_toRemote("Moving forward")
                key = 0

            data = "ef"
            

        elif data == "bc":            """Back"""
            print("back")
            
            if key == 1:
                so.send_toRemote("Moving back")
                key = 0

            cn.back()
            data = "ef"
            

        elif data == "cd":            """Right"""
            X = cp.captureImage()
            predict = (sess.run(tf.argmax(prediction,1), feed_dict={x: [X] } ))

            if predict == [2]:
                cn.right()
            else:
                selfDrive(predict)
                
            print("Right")
            
            if key == 1:
                so.send_toRemote("Moving right")
                key = 0
            
            data = "ef"
            
            

        elif data == "de":            """Left"""
            X = cp.captureImage()
            predict = (sess.run(tf.argmax(prediction,1), feed_dict={x: [X] } ))

            if predict == [0]:
                cn.left()
            else:
                selfDrive(predict)

            print("left")
            
            if key == 1:
                so.send_toRemote("Moving left")
                key = 0

            data = "ef"
            
            

        elif data == "ef":            """Stop"""
            print("stop")
            
            sig_ultrason = notice
            if sig_ultrason == 4:
                cn.forward_sonic()
                            
            if key == 1:
                so.send_toRemote("stop")
                key = 0

            cn.stop()


        elif data == "fg":            """End program"""
            msg = "end program"
            print("end program")
            
            prog_end()

            qw = 23
            q.put(qw)
            time.sleep(1)
            print("Terminating Thread 2")
                
            break



        else:
            """
            if a == 0:
                print("on loop")
                so.send_toRemote("on loop")
                a = 1
            """
            print("Within loop")
            so.send_toRemote("within loop")
                 
             
        print("Received message: ", data)    
                 


#prediction = neural_network_model(x)
#init = tf.global_variables_initializer()


q = Queue()

t1 = threading.Thread(target = sensor, args = (q,))
t2 = threading.Thread(target = modelCar, args = (q,))

t1.start()
t2.start()

t1.join()
t2.join()    


print("Program ends here")         
GPIO.cleanup()

