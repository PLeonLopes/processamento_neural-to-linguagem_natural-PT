import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from sklearn.datasets import make_circles
from numpy import where
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.callbacks import TensorBoard

tf.__version__

class ExtendedTensorBoard(TensorBoard):
    """
    Adaptado de https://github.com/tensorflow/tensorflow/issues/31542
    """
    def __init__(self, x, y,log_dir='logs',
                            histogram_freq=0,
                            write_graph=True,
                            write_images=False,
                            update_freq='epoch',
                            profile_batch=2,
                            embeddings_freq=0,
                            embeddings_metadata=None,
                            **kwargs,):
        self.x=x
        self.y=y
        super(ExtendedTensorBoard, self).__init__(log_dir,
                                                    histogram_freq,
                                                    write_graph,
                                                    write_images,
                                                    update_freq,
                                                    profile_batch,
                                                    embeddings_freq,
                                                    embeddings_metadata,)
    
    def _log_gradients(self, epoch):
        writer = self._get_writer(self._train_run_name)
        with writer.as_default(), tf.GradientTape() as g:
            
            features=tf.constant(self.x)
            y_true=tf.constant(self.y)
            
            y_pred = self.model(features)  # forward-propagation
            loss = self.model.compiled_loss(y_true=y_true, y_pred=y_pred)  # calculate loss
            gradients = g.gradient(loss, self.model.trainable_weights)  # back-propagation

            # In eager mode, grads does not have name, so we get names from model.trainable_weights
            for weights, grads in zip(self.model.trainable_weights, gradients):
                tf.summary.histogram(
                    weights.name.replace(':', '_') + '_grads', data=grads, step=epoch)

        writer.flush()

    def on_epoch_end(self, epoch, logs=None):
#         Sobre-escrevemos essa função da super classe pois necessitamos
#         adicionar a funcionalidade de gravar os gradientes.
#         Como tambem queremos suas funcionalidades originais, tambem invocamos o metodo super       
        super(ExtendedTensorBoard, self).on_epoch_end(epoch, logs=logs)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_gradients(epoch)



# DEFININDO OS DADOS

# gera dataset de classificação 
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)

# escala input para [-1,1]
scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)

# plota visualização do dataset
for i in range(2):
    samples_ix = where(y == i)
    plt.scatter(X[samples_ix, 0], X[samples_ix, 1], label=str(i))
plt.legend()
plt.show()

# separa em teste e treino
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]


def run_model(model,log_to_tb= False ,trainX=trainX,trainy=trainy,testX=testX,testy=testy):
    """
    Função auxiliar que recebe um modelo e realiza seu treinamento e avaliação no dataset.
    """
    model.summary()
    
    # compila modelo
    opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    #Cria log do modelo pra visualizarmos no TensorBoard se a flag estiver ligada
    callbacks = None
    if log_to_tb==True:
        log_dir = "logs/" + model.name
        callbacks=[ExtendedTensorBoard(x=trainX, y=trainy,log_dir=log_dir,histogram_freq=1)]
    

    # fit modelo
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=0, callbacks=callbacks)

    # avalia modelo
    _, train_acc = model.evaluate(trainX, trainy, verbose=0)
    _, test_acc = model.evaluate(testX, testy, verbose=0)
    print("\n")
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))


    # plota acurácia/training history
    plt.ylim(0, 1)
    plt.title("Acurácia "+ model.name)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()


# REDE RASA

#define modelo raso
init = RandomUniform(minval=0, maxval=1)

model = keras.Sequential(name="modelo_raso")
model.add(
    keras.layers.Dense(5,
    input_dim=2,
    activation="tanh",
    kernel_initializer=init,
    name="raso_1")
)

model.add(
    keras.layers.Dense(1,
    activation='sigmoid',
    kernel_initializer=init,
    name="raso_output")
)

run_model(model)


# REDE FUNDA

# define modelo mais fundo
init = RandomUniform(minval=0, maxval=1)

model = Sequential(name="modelo_fundo")
model.add(Dense(5, input_dim=2, activation='tanh', kernel_initializer=init,name="funda_1"))
model.add(Dense(5, activation='tanh', kernel_initializer=init,name="funda_2"))
model.add(Dense(5, activation='tanh', kernel_initializer=init,name="funda_3"))
model.add(Dense(5, activation='tanh', kernel_initializer=init,name="funda_4"))
model.add(Dense(5, activation='tanh', kernel_initializer=init,name="funda_5"))
model.add(Dense(1, activation='sigmoid', kernel_initializer=init,name="funda_output"))


run_model(model,log_to_tb=True)

# UTILIZAÇÃO DA INICIALIZAÇÃO DE XAVIER GLODOT

# define modelo mais fundo com inicializador de pesos melhor
model = Sequential(name="modelo_xavier")
model.add(Dense(5, input_dim=2, activation='tanh',kernel_initializer="glorot_uniform", name="xavier_1"))
model.add(Dense(5, activation='tanh',kernel_initializer="glorot_uniform", name="xavier_2"))
model.add(Dense(5, activation='tanh',kernel_initializer="glorot_uniform", name="xavier_3"))
model.add(Dense(5, activation='tanh', kernel_initializer="glorot_uniform",name="xavier_4"))
model.add(Dense(5, activation='tanh',kernel_initializer="glorot_uniform", name="xavier_5"))
model.add(Dense(1, activation='sigmoid', kernel_initializer="glorot_uniform", name="xavier_output"))

run_model(model,log_to_tb=True)
