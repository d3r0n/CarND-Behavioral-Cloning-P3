import csv
import os
import tensorflow as tf
import cv2

flags = tf.app.flags
FLAGS = flags.FLAGS
# command line flags
flags.DEFINE_string('src', '', "Srouce folder")

def read_lines(src = FLAGS.src + 'driving_log.csv'):
    lines = []
    with open(src) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)
    return lines

def main(_):
    lines = read_lines()
    print(lines[0])
    print(lines[2])

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
