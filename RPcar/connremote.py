import socket

"""socket settings"""
UDP_IP = "192.168.43.114"
UDP_PORT = 5005

UDP_IP_1 = "192.168.43.114"   # Receives data from Thread-Face
UDP_PORT_1 = 5004

TargetUDP_IP = "192.168.43.228"
TargetUDP_PORT = 5006



sock = socket.socket(socket.AF_INET, # Internet
socket.SOCK_DGRAM) # UDP

sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.5)   #1  0.5  0.3



sock_1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 

sock_1.bind((UDP_IP_1, UDP_PORT_1))
sock_1.settimeout(0.2)



conn = socket.socket(socket.AF_INET, # Internet
socket.SOCK_DGRAM) # UDP



def receive_fromRemote1():
    data_1, addr = sock_1.recvfrom(1024)
    data_1 = data_1.encode('utf-8')
    print("recv", data_1)
    return data_1

def receive_fromRemote():
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    data = data.encode('utf-8') # Actually should be decode() but don't know how it is working here
    print ("received message: ",  data)
    return data

def send_toRemote(msg):
    conn.sendto(msg.encode('utf-8'), (TargetUDP_IP, TargetUDP_PORT))


    
