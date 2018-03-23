import tensorflow as tf

from utils import mkdir_p
from PGGAN import PGGAN
from utils import CelebA
flags = tf.app.flags
import sys

flags.DEFINE_integer("OPER_FLAG", 0, "the flag of opertion: 0 is for training ")
flags.DEFINE_string("path" , 'train_images/', "the path of training data, for example /home/hehe/celebA/")
flags.DEFINE_integer("batch_size", 16, "batch size")
flags.DEFINE_integer("max_iters", 32000, "the maxmization of training number")
#flags.DEFINE_integer("max_iters", 2, "the maximum training iterations")
flags.DEFINE_float("learn_rate", 0.0001, "the learning rate for G and D networks")
flags.DEFINE_float("flag", 4, "the FLAG of gan training process")

FLAGS = flags.FLAGS
#if __name__ == "__main__":
#    main()
def main():
    root_log_dir = "./logs/"
    mkdir_p(root_log_dir)
    batch_size = FLAGS.batch_size
    max_iters = FLAGS.max_iters
    sample_size = 256
    GAN_learn_rate = FLAGS.learn_rate

    OPER_FLAG = FLAGS.OPER_FLAG
    data_In = CelebA(FLAGS.path)
    #print ("the num of dataset", len(data_In.image_list))

    if OPER_FLAG == 0:
        r_fl = 5
        test_list = [sys.argv[1]]
#'image_03902', 'image_06751', 'image_06069',
       #'image_05211', 'image_05757', 'image_05758',
       #'image_05105', 'image_03877', 'image_04325',
       #'image_05173', 'image_06667', 'image_03133',
       #'image_06625', 'image_06757', 'image_04065',
       #'image_03155'
        t = False
	f = open("captions_all.txt","a")
	f_c = open(sys.argv[2],"r")
	f.write(sys.argv[1]+'\n')
        for line in f_c:
            f.write(line)
	f.write("----\n")
	f.close()
	f_c.close()
        pggan_checkpoint_dir_write = "./model_flowers_test/"
        sample_path = "./PGGanFlowers/sample_test/"
        mkdir_p(pggan_checkpoint_dir_write)
        mkdir_p(sample_path)
        pggan_checkpoint_dir_read = "./model_flowers_{}/{}/".format(OPER_FLAG, r_fl)

        pggan = PGGAN(batch_size=batch_size, max_iters=max_iters,
                      model_path=pggan_checkpoint_dir_write, read_model_path=pggan_checkpoint_dir_read,
                      data=data_In, sample_size=sample_size,
                      sample_path=sample_path, log_dir=root_log_dir, learn_rate=GAN_learn_rate, PG= r_fl, t=t)

        pggan.build_model_PGGan()
        pggan.test(test_list,int(sys.argv[3]))

if __name__ == "__main__":
    main()
