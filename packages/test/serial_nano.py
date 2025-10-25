import serial, time # sudo apt install python3-serial
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2)

for d in (30, 90, 150):
    ser.write(f"S0={d}\n".encode())
    ser.flush()
    time.sleep(0.6)

ser.write(b"S1=us1500\n")
ser.flush()
