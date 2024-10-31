import numpy as np
import json

# Load data with LED positions
# led_data = json.load(open('led_array_pos_na_z65mm.json'))
led_data = json.load(open('led_array_distance_75.json'))
data = led_data['led_position_list_na']

# All patterns
total_patterns = []

# Pattern 1 (all BF for absorption)
low_na = 0.
high_na = 0.25
low_angle = -180
high_angle = 180
curr_patt = []
for led_nbr in data:
    curr_na = data[led_nbr]
    led_na = np.sqrt(curr_na[0]**2 + curr_na[1]**2)
    angle = np.angle(curr_na[0] + curr_na[1]*1j, deg=True)
    if led_na > low_na and led_na < high_na and angle >= low_angle and angle <= high_angle:
        curr_patt.append(int(led_nbr))
total_patterns.append(curr_patt)

# Pattern 2 (DPC phase 1)
low_na = 0
high_na = 0.25
low_angle = 0
high_angle = 180
curr_patt = []
for led_nbr in data:
    curr_na = data[led_nbr]
    led_na = np.sqrt(curr_na[0]**2 + curr_na[1]**2)
    angle = np.angle(curr_na[0] + curr_na[1]*1j, deg=True)
    if led_na > low_na and led_na < high_na and angle >= low_angle and angle <= high_angle:
        curr_patt.append(int(led_nbr))
total_patterns.append(curr_patt)

# Pattern 3 (DPC phase 2)
low_na = 0
high_na = 0.25
low_angle = -90
high_angle = 90
curr_patt = []
for led_nbr in data:
    curr_na = data[led_nbr]
    led_na = np.sqrt(curr_na[0]**2 + curr_na[1]**2)
    angle = np.angle(curr_na[0] + curr_na[1]*1j, deg=True)
    if led_na > low_na and led_na < high_na and angle >= low_angle and angle <= high_angle:
        curr_patt.append(int(led_nbr))
total_patterns.append(curr_patt)

# Pattern 4 bis (DPC phase 1 bis)
low_na = 0
high_na = 0.25
low_angle = -180
high_angle = 0
curr_patt = []
for led_nbr in data:
    curr_na = data[led_nbr]
    led_na = np.sqrt(curr_na[0]**2 + curr_na[1]**2)
    angle = np.angle(curr_na[0] + curr_na[1]*1j, deg=True)
    if led_na > low_na and led_na < high_na and angle >= low_angle and angle <= high_angle:
        curr_patt.append(int(led_nbr))
total_patterns.append(curr_patt)

# Pattern 5 bis (DPC phase 3 bis)  TODO phase 2?
low_na = 0
high_na = 0.25
low_angle = -90
high_angle = 90
curr_patt = []
for led_nbr in data:
    curr_na = data[led_nbr]
    led_na = np.sqrt(curr_na[0]**2 + curr_na[1]**2)
    angle = np.angle(curr_na[0] + curr_na[1]*1j, deg=True)
    if led_na > low_na and led_na < high_na and (angle <= low_angle or angle >= high_angle):
        curr_patt.append(int(led_nbr))
total_patterns.append(curr_patt)

# Pattern 6 (first ring of dark field)
low_na = 0.25
high_na = 0.25 * 1.3
low_angle = -180
high_angle = 180
curr_patt = []
for led_nbr in data:
    curr_na = data[led_nbr]
    led_na = np.sqrt(curr_na[0]**2 + curr_na[1]**2)
    angle = np.angle(curr_na[0] + curr_na[1]*1j, deg=True)
    if led_na > low_na and led_na < high_na and angle >= low_angle and angle <= high_angle:
        curr_patt.append(int(led_nbr))
total_patterns.append(curr_patt)

# Pattern 7 (same as 4 but removing a few LEDs for edge effect)
low_na = 0.26
high_na = 0.25 * 1.3
low_angle = -180
high_angle = 180
curr_patt = []
for led_nbr in data:
    curr_na = data[led_nbr]
    led_na = np.sqrt(curr_na[0]**2 + curr_na[1]**2)
    angle = np.angle(curr_na[0] + curr_na[1]*1j, deg=True)
    if led_na > low_na and led_na < high_na and angle >= low_angle and angle <= high_angle:
        curr_patt.append(int(led_nbr))
total_patterns.append(curr_patt)

# Pattern 8 (second ring of dark field)
low_na = 0.25 * 1.3
high_na = 0.25 * 1.7
low_angle = -180
high_angle = 180
curr_patt = []
for led_nbr in data:
    curr_na = data[led_nbr]
    led_na = np.sqrt(curr_na[0]**2 + curr_na[1]**2)
    angle = np.angle(curr_na[0] + curr_na[1]*1j, deg=True)
    if led_na > low_na and led_na < high_na and angle >= low_angle and angle <= high_angle:
        curr_patt.append(int(led_nbr))
total_patterns.append(curr_patt)

# Pattern 9 (third ring of dark field)
low_na = 0.25 * 1.7
high_na = 0.25 * 2
low_angle = -180
high_angle = 180
curr_patt = []
for led_nbr in data:
    curr_na = data[led_nbr]
    led_na = np.sqrt(curr_na[0]**2 + curr_na[1]**2)
    angle = np.angle(curr_na[0] + curr_na[1]*1j, deg=True)
    if led_na > low_na and led_na < high_na and angle >= low_angle and angle <= high_angle:
        curr_patt.append(int(led_nbr))
total_patterns.append(curr_patt)

# Pattern 10 (DF-PPR with fewer images = larger rings)
low_na = 0.26
high_na = 0.25 * 1.5
low_angle = -180
high_angle = 180
curr_patt = []
for led_nbr in data:
    curr_na = data[led_nbr]
    led_na = np.sqrt(curr_na[0]**2 + curr_na[1]**2)
    angle = np.angle(curr_na[0] + curr_na[1]*1j, deg=True)
    if led_na > low_na and led_na < high_na and angle >= low_angle and angle <= high_angle:
        curr_patt.append(int(led_nbr))
total_patterns.append(curr_patt)

# Pattern 11 (DF-PPR with fewer images = larger rings)
low_na = 0.25 * 1.5
high_na = 0.25 * 2
low_angle = -180
high_angle = 180
curr_patt = []
for led_nbr in data:
    curr_na = data[led_nbr]
    led_na = np.sqrt(curr_na[0]**2 + curr_na[1]**2)
    angle = np.angle(curr_na[0] + curr_na[1]*1j, deg=True)
    if led_na > low_na and led_na < high_na and angle >= low_angle and angle <= high_angle:
        curr_patt.append(int(led_nbr))
total_patterns.append(curr_patt)

# Save total_patterns as json file

# Serializing json
json_object = json.dumps(total_patterns, indent=4)

# Writing to sample.json
with open("sample.json", "w") as outfile:
    outfile.write(json_object)
