import time
import Jetson.GPIO as GPIO

PIN = 32
FREQ_HZ = 100

GPIO.setmode(GPIO.BOARD)
GPIO.setup(PIN, GPIO.OUT, initial=GPIO.LOW)
pwm = GPIO.PWM(PIN, FREQ_HZ)

def angle_to_us(angle, us_min=50, us_max=2200):
    angle = max(0.0, min(360.0, float(angle)))
    return us_min + (us_max - us_min) * (angle / 360.0)

def us_to_duty(us, freq_hz=FREQ_HZ):
    period_us = 1_000_000.0 / freq_hz
    return max(0.0, min(100.0, 100.0 * (us / period_us)))

start_us = angle_to_us(180)
pwm.start(us_to_duty(start_us))

try:
    for a in range(0, 361, 30):
        pulse_us = angle_to_us(a)
        duty = us_to_duty(pulse_us)
        pwm.ChangeDutyCycle(duty)
        time.sleep(0.2)
finally:
    pwm.stop()
    GPIO.cleanup()
