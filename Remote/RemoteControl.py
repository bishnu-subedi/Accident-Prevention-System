"""Remote for the project. Controlled by user"""

from tkinter import*
import socket

import threading, queue
import time, datetime

import numpy as np
import tensorflow as tf

import cv2


font = cv2.FONT_HERSHEY_SIMPLEX


size = 2
reserve = np.arange(1, 3201)
cap = cv2.VideoCapture(0) #Use camera 0

#loading the xml file
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


model_path_obj = "C:\\Users\\Dell\\Desktop\\Projects\\majorproj\\faceMatching\\model_trained\\modeluserobj5\\modeluser-obj-5"
model_path_eye = "C:\\Users\\Dell\\Desktop\\Projects\\majorproj\\faceMatching\\model_trained\\modelusereye5\\modeluser-eye-5"



"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""


UDP_IP = "192.168.43.228"    #Remote Pc ip
UDP_PORT = 5006             #Remote pc port

TargetUDP_IP = "192.168.43.115"   #Server ip
TargetUDP_PORT = 5005             #Server ip

TargetUDP_IP_1 = "192.168.43.115"   #Server ip
TargetUDP_PORT_1 = 5004             #Server ip


print ("UDP target IP:", TargetUDP_IP)
print ("UDP target port:", TargetUDP_PORT)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Internet, UDP

conn = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Internet, UDP
conn.bind((UDP_IP, UDP_PORT))   #For receiving purpose


messg = "received message: "
messg_1 = "sent: "




class Win():
    
    def __init__(self, master):
        # the container is where we'll stack a bunch of frames on top of each other,
        # then the one we want visible will be raised above the others
        container = Frame(master)
        self.master = master
        container.pack(side="top", fill="both", expand= True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        

        self.frames = {}

        for F in ( FrontPage, MainPage):
            frame = F(parent=container, controller=self)
            self.frames[F] = frame
            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(MainPage)

        

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()




class MainPage(Frame):

    def receivefromServer(self):
        try:
            data, addr = conn.recvfrom(1024)
            data = data.decode('utf-8')
            #print ("received msg:", data)
        except Exception:
            data = "nothing.."

        self.labl = Label(self, text=messg)
        self.labl.grid(row=7, column=3)
        self.lab2 = Label(self, text=data)
        self.lab2.grid(row=7, column=4)
        


    def sendtoServer(self, MESSAGE):
        
        #print("sent: ", MESSAGE)
        self.lab3 = Label(self, text=messg_1)
        self.lab3.grid(row=8, column=3)
        self.lab4 = Label(self, text=MESSAGE)
        self.lab4.grid(row=8, column=4)
        
        sock.sendto(MESSAGE.encode("utf-8"), (TargetUDP_IP, TargetUDP_PORT))
        #receivefromServer()




    def toggle(self):

        if self.button7.config('text')[-1] == 'OFF':
            self.button7.config(text='ON')
            #print ("pressed true")
            self.sendtoServer("man")
            
        else:
            self.button7.config(text='OFF')
            self.sendtoServer("man")

        #self.sendtoServer("manual")
    


    
    def __init__(self, parent, controller):
        Frame.__init__( self, parent)

        self.controller = controller

        #c = input("enter:")
        self.receivefromServer()

        print("main")


        self.lb = Label(self, text = "Front Page")
        self.lb1 = Label(self, text = " ")
        #self.button = Button(self, text="Forward", command = lambda: self.controller.show_frame(Page_One))
        self.button1 = Button(self, text="Forward", command = lambda: self.sendtoServer("ab"))
        self.button2 = Button(self, text="Back", command = lambda: self.sendtoServer("bc"))
        self.button3 = Button(self, text="Right", command = lambda: self.sendtoServer("cd"))
        self.button4 = Button(self, text="Left", command = lambda: self.sendtoServer("de"))
        self.button5 = Button(self, text="Stop", command = lambda: self.sendtoServer("ef"))
        self.button6 = Button(self, text="END program", command = lambda: self.sendtoServer("fg"))
        self.button7 = Button(self, text="OFF", width=12, command = lambda: self.toggle())

        self.lb.grid(row = 0, column = 3)
        self.lb1.grid(row = 1, column = 2)
        
        self.button1.grid(row = 3, column = 4)
        self.button2.grid(row = 4, column = 4)
        self.button3.grid(row = 4, column = 7)
        self.button4.grid(row = 4, column = 3)
        self.button5.grid(row = 3, column = 5)
        self.button6.grid(row = 5, column = 5)
        self.button7.grid(row = 6, column = 3)




class FrontPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__( self, parent)

        self.controller = controller
        
        #self.m = Button(self, text="OK", command = lambda: controller.show_frame(StartPage))
        #self.m.grid(row = 1)

        print("front")

        self.lbl = Label(self, text=" User Eyes are closed or Phone detected ")
        self.lbl.grid(row = 0, column = 3)

        

"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""



def rgb2gray_eye(rgb):
    """rgb = cv2.resize(rgb, (40, 80))
    print('Gray shape:', rgb.shape)
    """
    rgb = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    try:
        rgb = cv2.resize(rgb, (40, 80))  #80, 40 prevoiusly
    except Exception:
        #print("Error")
        return 0 
    #print('Gray shape:', rgb.shape)
    #cv2.imwrite('gray.jpg', rgb)
    b = np.asarray(rgb)
    b = b.flatten()
    X = b.astype(int)
    predict_run = (sess1.run(tf.argmax(prediction_eye,1), feed_dict={x: [X] } ))
    #print(predict_run)

    
    """#return int(predict_run.flatten() != 1)"""
    return predict_run.flatten()



def rgb2gray_obj(rgb):
    
    rgb = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    try:
        rgb = cv2.resize(rgb, (40, 80))  #80, 40 prevoiusly
    except Exception:
        #print("Error")
        return 0 
    #print('Gray shape:', rgb.shape)
    #cv2.imwrite('gray.jpg', rgb)
    b = np.asarray(rgb)
    b = b.flatten()
    X = b.astype(int)
    predict_run = (sess2.run(tf.argmax(prediction_obj,1), feed_dict={x_1: [X] } ))
    #print(predict_run)

    
    return predict_run.flatten()



"""Defining model"""
def neural_network_model(data):
    #(input_data * weights) + biases
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([3200, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    #hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
    #                  'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}


    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.sigmoid(l1)
                        
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.sigmoid(l2)

    output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']

    return output



def neural_network_model_1(data):
    #(input_data * weights) + biases
    hidden_1_layer_1 = {'weights': tf.Variable(tf.random_normal([3200, n_nodes_hl1_1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1_1]))}

    hidden_2_layer_1 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1_1, n_nodes_hl2_1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2_1]))}
    
    #hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
    #                  'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    output_layer_1 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2_1, n_classes_1])),
                      'biases': tf.Variable(tf.random_normal([n_classes_1]))}


    l1_1 = tf.add(tf.matmul(data, hidden_1_layer_1['weights']), hidden_1_layer_1['biases'])
    #rectified linear: relu
    #l1 = tf.nn.relu(l1)  #Not accurate or perfect as sigmoid
    #l1 = tf.nn.softmax(l1)  #Not very good either
    #l1_1 = tf.nn.sigmoid(l1_1)
    l1_1 = tf.nn.tanh(l1_1)
                        
    l2_1 = tf.add(tf.matmul(l1_1, hidden_2_layer_1['weights']), hidden_2_layer_1['biases'])
    #rectified linear: relu
    #l2 = tf.nn.relu(l2)
    #l2_1 = tf.nn.sigmoid(l2_1)
    l2_1 = tf.nn.tanh(l2_1)

    output_1 = tf.matmul(l2_1, output_layer_1['weights']) + output_layer_1['biases']

    return output_1


"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""



def face(num):           """Thread 2"""
    root = Tk()
    root.title(" Accident Prevention System ")
    root.geometry("500x500")
    print("remote started")
    app = Win(root)
    root.after(2000)
    root.mainloop()



def tkRemote(num):       """Thread 1"""
    """#global alt"""
    left = right = retina = 0
    count = count_1 = 0

    ent = "close"

    while True:
        (rval, im) = cap.read()
        # Resizing image to decrease computation
        mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

        # detecting faces 
        faces = classifier.detectMultiScale(mini)

        # Drawing rectangles around each face
        for f in faces:
            (fx, fy, fw, fh) = [v * size for v in f] #Scale the shapesize backup
            cv2.rectangle(im, (fx, fy), (fx + fw, fy + fh),(0,255,0),thickness=4)

            sub_face = im[fy+30:fy+fh, fx:fx+60]
            right = rgb2gray_obj(sub_face)
                    
            #print(sub_face.shape)
            #print(X)
                    
            #predict_run = (sess.run(tf.argmax(prediction,1), feed_dict={x: [X] } ))
            #print(predict_run)
                    
            sub_face1 = im[fy+30:fy+fh, fx+180:fx+fw]   #x+140:x+w
            left = rgb2gray_obj(sub_face1)

            #predict_run = (sess.run(tf.argmax(prediction,1), feed_dict={x: [X] } ))
            #print(predict_run)
            
            
            if((left or right) == 1):
                #print("phone Detected")
                cv2.putText(im, "Phone Detected!", (10, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                ent = "close"
                sock.sendto(ent.encode("utf-8"), (TargetUDP_IP_1, TargetUDP_PORT_1))
                #count = count + 1
                #sock.sendto(ent.encode("utf-8"), (TargetUDP_IP_1, TargetUDP_PORT_1))
            """else:
                mnb = 1
                #print("object 'NOT' Detected")
                #count_1 = count_1 + 1          """
                
        eyes = eye_cascade.detectMultiScale(mini)
        for e in eyes:
            (ex, ey, ew, eh) = [ve * size for ve in e] #Scale the shapesize backup
            cv2.rectangle(im, (ex, ey), (ex + ew, ey + eh),(255,0,0),thickness=1)

            sub_face2 = im[ey:ey+eh, ex:ex+ew]
            retina = rgb2gray_eye(sub_face2)


            if(retina == 0):
                #print("eye_closed")
                cv2.putText(im, "close", (400, 30), font, 0.8, (255,0,0), 2, cv2.LINE_AA)
                #count = count + 1
                ent = "close"
                sock.sendto(ent.encode("utf-8"), (TargetUDP_IP_1, TargetUDP_PORT_1))
            else:
                #print("eye_open")
                cv2.putText(im, "open", (470, 30), font, 0.8, (255,0,0), 2, cv2.LINE_AA)
                #count_1 = count_1 + 1
                #ent = "open"
                #sock.sendto(ent.encode("utf-8"), (TargetUDP_IP_1, TargetUDP_PORT_1))
                
 
        # Show the image
        cv2.imshow('Detecting User',   im)

        """If 'q' entered then program ends."""
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break

        
    cap.release()
    cv2.destroyAllWindows()


    
"""<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"""
#t = time.time()
arg = [1,2,3]



g_1 = tf.Graph()
with g_1.as_default():
    n_nodes_hl1 = 280  #260 modelobj1
    n_nodes_hl2 = 80   #50 modelobj1
    #n_nodes_hl3 = 500

    n_classes = 2      #no. of labels

    #tf.reset_default_graph()

    """40x80 pixels image = flat 3200 pixels(features of each example in data)"""
    x = tf.placeholder('float', [None, 3200])
    y = tf.placeholder('float')

    prediction_eye = neural_network_model(x)

    saver1 = tf.train.Saver()


sess1 = tf.Session(graph = g_1)
#sess1.run(init)
saver1.restore(sess1, model_path_eye)



g_2 = tf.Graph()
with g_2.as_default():
    n_nodes_hl1_1 = 280  #260 modelobj1
    n_nodes_hl2_1 = 80   #50 modelobj1
    #n_nodes_hl3 = 500

    n_classes_1 = 2      #no. of labels

    #tf.reset_default_graph()

    """40x80 pixels image = flat 3200 pixels(features of each example in data)"""
    x_1 = tf.placeholder('float', [None, 3200])
    y_1 = tf.placeholder('float')

    prediction_obj = neural_network_model_1(x_1)

    saver2 = tf.train.Saver()


sess2 = tf.Session(graph = g_2)
#sess1.run(init)
saver2.restore(sess2, model_path_obj)


t1 = threading.Thread(target = face, args = (arg,))
t2 = threading.Thread(target = tkRemote, args = (arg,))

t1.start()
t2.start()

t1.join()
t2.join()

print("program exited......")

