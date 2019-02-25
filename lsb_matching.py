# coding: utf-8

import numpy as np
import tensorflow as tf
from PIL import Image
#import texts
import logging
import random
import string

KEY_LEN = 10

def base_str():
    return (string.ascii_letters +string.digits)
def key_gen():
    keylist = [random.choice(base_str()) for i in range(KEY_LEN)]
    return ("".join(keylist))

#texts = 'qqqqqqqqqq'
texts = key_gen()
def str_to_bin(text):
    bins = ''.join(format(x, 'b') for x in bytearray(text, 'utf8'))
    return np.array(list(bins), dtype=np.int32)
'''
class BaseStego:
    DELIMITER = np.ones(100, dtype=int)  # TODO hidden info ends with 1, then decoder skip it

    def __init__(self):
        pass


    def get_information(batch_size=64, len_of_text=500):
        start_idxes = np.random.randint(0, len(texts) - 1, batch_size)
        return [str_to_bin(texts[start_idx:start_idx+len_of_text]) for start_idx in start_idxes]
'''

class LSBMatching():
    def __init__(self):
        pass

    @staticmethod
    def get_information(batch_size=64, len_of_text=500):
        start_idxes = np.random.randint(0, len(texts) - 1, batch_size)
        return [str_to_bin(texts[start_idx:start_idx+len_of_text]) for start_idx in start_idxes]
    @staticmethod
    def tf_encode(container):
        """
        LSB matching algorithm (+-1 embedding)
        :param container: tf tensor shape (batch_size, width, height, chan)
        :param information: array with int bits
        :param stego: name of image with hidden message
        """
        with tf.variable_scope('Stego'):

            n, width, height, chan = 64, 64, 64, 3
            logging.debug('batch_n', n)

            information = LSBMatching.get_information(n, 10)
            logging.debug('Information to hide', information)

            mask = np.zeros(list((n, width, height, chan)))

            print('Num of images: %s' % n)
            for img_idx in range(n):
                print(img_idx)

                for i, bit in enumerate(information[img_idx]):
                    ind, jnd = i // width, i - width * (i // width)

                    if tf.to_int32(container[img_idx, ind, jnd, 0]) % 2 != bit:
                        if np.random.randint(0, 2) == 0:
                            # tf.assign_sub(container[img_idx, ind, jnd, 0], 1)
                            mask[img_idx, ind, jnd, 0] += 1
                        else:
                            # tf.assign_add(container[img_idx, ind, jnd, 0], 1)
                            mask[img_idx, ind, jnd, 0] -= 1

            #logger.debug('Finish encoding')
            return tf.add(container, mask)

    @staticmethod
    def encode(container, information, stego='stego.png'):
        """
        LSB matching algorithm (+-1 embedding)
        :param container: path to image container
        :param information: array with int bits
        :param stego: name of image with hidden message
        """
        img = Image.open(container)
        width, height = img.size
        img_matr = np.asarray(img)
        img_matr.setflags(write=True)

        red_ch = img_matr[:, :, 2].reshape((1, -1))[0]

        information = np.append(information, np.ones(100, dtype=int))
        for i, bit in enumerate(information):

            if bit != red_ch[i] & 1:
                if np.random.randint(0, 2) == 0:
                    red_ch[i] -= 1
                else:
                    red_ch[i] += 1

        img_matr[:, :, 2] = red_ch.reshape((height, width))

        Image.fromarray(img_matr).save(stego)

    @staticmethod
    def decode(container):
        img = Image.open(container)
        img_matr = np.asarray(img)

        red_ch = img_matr[:, :, 2].reshape((1, -1))[0]

        delim_len = len(np.ones(100, dtype=int))

        info = np.array([], dtype=int)
        for pixel in red_ch:
            info = np.append(info, [pixel & 1])

            if info.shape[0] > delim_len and np.array_equiv(info[-delim_len:], np.ones(100, dtype=int)):
                break

        info = info[:-delim_len]

        return ''.join(map(str, info))
