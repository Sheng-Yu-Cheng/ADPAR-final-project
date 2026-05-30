from djitellopy import Tello
import time


# ============================================================
# Tello square flight path for radar visibility
#
# Path:
#   takeoff
#   move up
#   forward
#   right
#   back
#   left
#   land
#
# Radar suggestion:
#   Put radar facing the drone.
#   The forward/back segments should be most visible in range-Doppler.
# ============================================================

tello = Tello()
tello.connect()

print("Battery:", tello.get_battery())

# Safety check
if tello.get_battery() < 30:
    print("Battery too low. Please charge before flight.")
    tello.end()
    raise SystemExit

try:
    # Take off
    tello.takeoff()
    time.sleep(2)

    # Move to a visible height.
    # 50 cm is usually okay indoors.
    tello.move_up(50)
    time.sleep(1)

    # Square size in cm.
    # Start conservative; increase to 120~150 cm if the space is clear.
    SIDE_CM = 100

    # Pause after each move so radar frames can clearly capture each segment.
    PAUSE_S = 1.0

    print("Flying square: forward -> right -> back -> left")

    # 1. Move away / toward radar depending on radar placement
    tello.move_forward(SIDE_CM)
    time.sleep(PAUSE_S)

    # 2. Move laterally right
    tello.move_right(SIDE_CM)
    time.sleep(PAUSE_S)

    # 3. Move backward
    tello.move_back(SIDE_CM)
    time.sleep(PAUSE_S)

    # 4. Move laterally left, roughly back to start
    tello.move_left(SIDE_CM)
    time.sleep(PAUSE_S)

    print("Square complete. Landing.")

finally:
    # Always try to land safely.
    tello.land()
    tello.end()