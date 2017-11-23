import tensorflow as tf


def main():
    hello = tf.constant('HelloWorld')
    with tf.Session() as sess:
        print(sess.run(hello))


if __name__ == '__main__':
    main()
