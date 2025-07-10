"""
Simple line tracking test example.
This example uses the grayscale sensors to track a line on the ground.
It moves forward when the line is detected in the center, turns left when the line is on the left, and turns right when the line is on the right.
"""

import time
import picar_4wd as fc

TRACK_LINE_SPEED = 20


def track_line():
    gs_list = fc.get_grayscale_list()
    status = fc.get_line_status(400, gs_list)
    if status == 0:
        fc.forward(TRACK_LINE_SPEED)
    elif status == -1:
        fc.turn_left(TRACK_LINE_SPEED)
    elif status == 1:
        fc.turn_right(TRACK_LINE_SPEED)


if __name__ == "__main__":
    try:
        while True:
            track_line()
            # give the ADC a moment before the next read
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        fc.stop()
        print("Program stop")
