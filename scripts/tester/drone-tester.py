from djitellopy import Tello
import time

tello = Tello()
tello.connect()

print("Battery:", tello.get_battery())

tello.takeoff()
time.sleep(1)

tello.move_up(50)
tello.move_forward(100)
tello.rotate_clockwise(90)
tello.move_back(100)

tello.land()