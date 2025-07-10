"""
Test picar-4wd to move forward for 1 second.
This is a simple example to demonstrate the basic movement functionality of the car.
"""

import picar_4wd as fc
import time

try:
    while True:
        fc.forward(50)
        time.sleep(1)
finally:
    fc.stop()
    time.sleep(0.2)