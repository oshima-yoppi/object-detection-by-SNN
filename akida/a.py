import tensorflow as tf
print(tf.__version__)
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
print( tf.matmul(matrix1, matrix2) )
exit()