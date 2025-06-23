"""Simple line tracking example.

The original example performed no delay between iterations and called
``get_line_status`` multiple times which resulted in unnecessary I2C traffic.
This version caches the sensor reading, adds a tiny delay to give the bus some
breathing room and improves readability.
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
