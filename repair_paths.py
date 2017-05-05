import csv
import os
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
# command line flags
flags.DEFINE_string('src', '', "Srouce folder")

slash = '/'
if os.name == 'nt':
    slash = '\\'


def main(_):
    lines = []
    src = FLAGS.src + 'driving_log.csv'
    bcp = FLAGS.src + 'driving_log_bcp.csv'
    os.rename(src, bcp)
    with open(bcp) as A, open(src, 'w+') as B:
        reader = csv.reader(A)
        writer = csv.writer(B)
        for line in reader:
            for i in range(3):
                line[i] = "input/IMG/" + line[i].split(slash)[-1]
            writer.writerow(line.rstrip())


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
