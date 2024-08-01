from program import program
from model import train_model, valid_model
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('MODE', 'program','Set program to run in different mode, include train, valid and program.')
flags.DEFINE_string('checkpoint_dir', './ckpt', 'Path to model file.')
flags.DEFINE_string('train_data', './data/fer2013/fer2013.csv','path to training data.')
flags.DEFINE_string('valid_data', './valid_sets/','Path to training data.')
flags.DEFINE_string('show box', False, 'If true, the results will show detection box')
FLAGS = flags.FLAGS

def main():
    assert FLAGS.MODE in ('train', 'valid', 'program')

    if FLAGS.MODE == 'program':
        program(FLAGS.checkpoint_dir, FLAGS.show_box)
    elif FLAGS.MODE == 'train':
        train_model(FLAGS.train_data)
    elif FLAGS.MODE == 'valid':
        valid_model(FLAGS.checkpoint_dir, FLAGS.valid_data)

if __name__ == '__main__':
    main()

