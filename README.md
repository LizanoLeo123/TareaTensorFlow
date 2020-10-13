# Tarea TensorFlow

Tarea de Semana 6 del curso de Introducción al reconocimiento de patrones
By Leonardo Lizano

El código del modelo es el siguiente

import tensorflow as tf

num_classes = 10
img_rows, img_cols = 28, 28
num_channels = 1
input_shape = (img_rows, img_cols, num_channels)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

## Creating the model

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

#Training the model
model.compile(optimizer='sgd',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(x_train, y_train, epochs=5, verbose=1, validation_data=(x_test, y_test))

## Preguntas

¿Qué es Keras comparado con TensorFlow, y cuál es su función?
Son sistemas con una arquitectua similar que conviven uno con el otro. TensorFlow es una arquitectura que permite hacer deep learning mediante el uso de Tensors, que es un tipo de dato propio, Keras se usa para crear y entrenar los modelos.

¿Por qué TensorFlow uasr "graphs" o gráficos y como se crean manualmente?
Un Tensor es un arreglo de N-dimensiones, los gráficos son una forma de poder representarlos y mostrarlos.
En TensworFlow 2 los gráficos ya no los tiene que hacer el usuario.

¿Cuál es la diferencia entre el modo de 'eager execution' y 'lazy execution'?
TensorFlow2 implementa un modo llamado "Eager Execution", la diferencia con su predecesor que siempre usaba "Lazy Exceution" por defecto. Es que las operaciones no se ejecutban por el framework si no se le indicaba específicamente que fuese así.
Este cambio lo que hace es optimizar algunos resultados y datos de salida.

¿Cómo ingresar información en TensorBoard y como mostrarla?
TensorBoard es una herramienta para mostrar la información de los procesos de TensowFlow de manera más detallada.
La información de los datos a analizar se ingresa a modo de callbacks, se ingresan los datos y un folder de salida, de forma que al hacer el model.fit, TensorBoard crea los gráficos de salida en el folder indicado.

Ejemplo:
callbacks = [tf.keras.callbacks.TensorBoard('./logs_keras')]
model.fit(x_train, y_train, epochs=5, verbose=1, validation_data=(x_test, y_test), callbacks=callbacks)

¿Cuál es la diferencia principal entre TensorFlow1 y TensorFlow2?
Funcionalidad, al igual que el manejo de los datos a nivel de creación del modelo, como también de la representación de los mismos.
