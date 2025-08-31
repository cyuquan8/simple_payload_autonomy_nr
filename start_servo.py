import math, time
from dronekit import connect
from rpi_hardware_pwm import HardwarePWM

# Servo setup
pwm = HardwarePWM(0, 50)  # 50 Hz for servo
pwm.start(5)            # neutral (0 degrees)

connection_string = '/dev/ttyAMA0'
print(f"Connecting to vehicle on: {connection_string}")
vehicle = connect(connection_string, wait_ready=True, baud=57600)
print("Connected!")

base_angle = 35  # servo neutral position (camera tilted)

try:
    while True:
        pitch = math.degrees(vehicle.attitude.pitch)
        print(f"Pitch: {pitch:.2f} degrees")

        # Compensate
        servo_angle = base_angle - pitch  

        # Limit servo movement
        servo_angle = max(min(servo_angle, 90), 0)

        # Convert angle to duty cycle
        duty_cycle = 5 + (servo_angle / 90) * 5  

        print(f"Servo angle: {servo_angle:.1f}°, Duty: {duty_cycle:.2f}%")
        pwm.change_duty_cycle(duty_cycle)

        time.sleep(0.02)  # 20 ms update
except KeyboardInterrupt:
    vehicle.close()
    pwm.stop()
    print("Stopped and closed.")
