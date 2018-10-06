import RPi.GPIO as GPIO

"""GPIO settings"""
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)

GPIO.setup(11, GPIO.OUT)
GPIO.setup(12, GPIO.OUT)
GPIO.setup(13, GPIO.OUT)

GPIO.setup(15, GPIO.OUT)
GPIO.setup(16, GPIO.OUT)
GPIO.setup(18, GPIO.OUT)

my_pwm = GPIO.PWM(13, 8)  #10#100 Hz
my_pwm.start(0)
my_pwm1 = GPIO.PWM(18, 8)  #10#100 Hz
my_pwm1.start(0)



"""GPIO Function"""
def forward_sonic():
    GPIO.output(11, False)     
    GPIO.output(12, True)

    my_pwm.ChangeDutyCycle(40)
    #sleep(0.04)

    GPIO.output(15, False)
    GPIO.output(16, True)

    my_pwm1.ChangeDutyCycle(40)
    sleep(0.08)


    

def forward():
    GPIO.output(11, False)     
    GPIO.output(12, True)

    my_pwm.ChangeDutyCycle(15) #30
    sleep(0.04)

    GPIO.output(15, False)
    GPIO.output(16, True)

    my_pwm1.ChangeDutyCycle(15)  #30
    sleep(0.04)

    """my_pwm.stop()"""



def back():
    
    GPIO.output(11, True)
    GPIO.output(12, False)

    my_pwm.ChangeDutyCycle(30)
    sleep(0.04)

    GPIO.output(15, True)
    GPIO.output(16, False)

    my_pwm1.ChangeDutyCycle(30)    
    sleep(0.04)
    """my_pwm.stop()"""


def right():
    GPIO.output(11, True)
    GPIO.output(12, False)

    my_pwm.ChangeDutyCycle(82) #62
    #sleep(0.04)

    GPIO.output(15, False)
    GPIO.output(16, True)

    my_pwm1.ChangeDutyCycle(30) #20
    #sleep(0.0003)
    """my_pwm.stop()"""
    

def left():
    GPIO.output(11, False)
    GPIO.output(12, True)

    my_pwm.ChangeDutyCycle(95) #75#30 #55
        #sleep(0.04)

    GPIO.output(15, True)
    GPIO.output(16, False)

    my_pwm1.ChangeDutyCycle(40)  #30#10

    sleep(0.04)
        
    """my_pwm.stop()"""
    #sleep(0.0003)
    

def stop():

    GPIO.output(11, False)
    GPIO.output(12, False)

    my_pwm.ChangeDutyCycle(0)
    sleep(0.04)

    GPIO.output(15, False)
    GPIO.output(16, False)

    my_pwm1.ChangeDutyCycle(0)
    sleep(0.04)
    """my_pwm.stop()"""


def prog_end():
    """Ending program"""

    my_pwm.stop()
    my_pwm1.stop()
    GPIO.cleanup()
