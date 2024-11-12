import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy.random import seed
import keras as ke
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from tensorflow.python.framework.random_seed import set_random_seed

import argparse
parser = argparse.ArgumentParser(description='Process a input file name.')

parser.add_argument('filename', type=str, help='The name of the file to process.')
parser.add_argument('dir1', type=str, help='The output directory path.')
args = parser.parse_args()

input_file= args.filename
output_path = args.dir1
print(output_path)

print(f"sklearn: {sk.__version__}")
print(f"pandas: {pd.__version__}")
print(f"keras: {ke.__version__}")

#Read input data
#ExpressionData.csv if you want to generate expression vector. If you want to generate motif vector, enter gene_motif_matrix.csv
input_df = pd.read_csv(input_file, sep = ',', index_col = 0)
input_df = input_df.astype('float16')
input_df = pd.DataFrame(input_df.values, index = input_df.index.astype(str))
X = input_df

#Split into train/test set
X_train, X_test = train_test_split(X, test_size=0.2, random_state=123)

#Standardize the data
scaler = StandardScaler().fit(X_train)
scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
X_train = X_train.pipe(scale_df, scaler)
X_test = X_test.pipe(scale_df, scaler)

# Class for autoencoder
class Autoencoder(object):

    def __init__(self, n_features, latent_dim, random_seed):

        # Set random seeds
        seed(123456 * random_seed)
        set_random_seed(123456 * random_seed)

        # Define latent size
        self.latent_dim = latent_dim

        # Define input layer
        ae_inputs = Input(shape=(n_features,))

        # Define autoencoder net
        [ae_net, encoder_net, decoder_net] = self._create_autoencoder_net(ae_inputs, n_features, latent_dim)
        print("AE net ")
        ae_net.summary()
        print("Encoder net ")
        encoder_net.summary()
        print("Decoder net ")
        decoder_net.summary()

        # compile models
        self._trainable_ae_net = self._make_trainable(ae_net)
        self._ae = self._compile_ae(ae_net)
        self._encoder = self._compile_encoder(encoder_net)
        self._decoder = self._compile_decoder(decoder_net)

    def _make_trainable(self, net):
        def make_trainable(flag):
            net.trainable = flag
            for layer in net.layers:
                layer.trainable = flag

        return make_trainable

    # Method for defining autoencoder network
    # Take input features, generate z encoding
    # Reconstruct the data
    def _create_autoencoder_net(self, inputs, n_features, latent_dim):

        # Encoder
        dense1 = Dense(700, activation='relu')(inputs)
        dropout1 = Dropout(0.1)(dense1)
        latent_layer = Dense(latent_dim)(dropout1)

        # Decoder
        dense2 = Dense(700, activation='relu')
        dropout2 = Dropout(0.1)
        outputs = Dense(n_features)

        decoded = dense2(latent_layer)
        decoded = dropout2(decoded)
        decoded = outputs(decoded)

        autoencoder = Model(inputs=[inputs], outputs=[decoded])
        encoder = Model(inputs=[inputs], outputs=[latent_layer])

        # Define decoder
        decoder_input = Input(shape=(latent_dim,))
        decoded = dense2(decoder_input)
        decoded = dropout2(decoded)
        decoded = outputs(decoded)
        decoder = Model(inputs=decoder_input, outputs=[decoded])

        return [autoencoder, encoder, decoder]

    # Compile model
    def _compile_ae(self, ae_net):
        ae = ae_net
        self._trainable_ae_net(True)
        ae.compile(loss='mse', metrics=['mse'], optimizer=Adam(learning_rate=0.00001))
        return ae

    # Compile model
    def _compile_encoder(self, encoder_net):
        ae = encoder_net
        self._trainable_ae_net(True)
        ae.compile(loss='mse', metrics=['mse'], optimizer=Adam(learning_rate=0.00001))
        return ae

    # Compile model
    def _compile_decoder(self, decoder_net):
        ae = decoder_net
        self._trainable_ae_net(True)
        ae.compile(loss='mse', metrics=['mse'], optimizer=Adam(learning_rate=0.00001))
        return ae

    #Training
    def fit(self, x, validation_data=None, T_iter=250, batch_size=128):
        if validation_data is not None:
            x_val = validation_data
        # Record training and validation metrics
        self._train_metrics = pd.DataFrame()
        self._val_metrics = pd.DataFrame()

        # Go over all iterations
        for idx in range(T_iter):
            print("Iter ", idx)

            # train classifier
            self._trainable_ae_net(True)
            history = self._ae.fit(x, x,
                                   batch_size=batch_size, epochs=1, verbose=1,
                                   validation_data=(x_val, x_val))

            print("Autoencoder loss ", history.history)
            self._train_metrics.loc[idx, 'AE loss'] = history.history['loss'][0]
            self._val_metrics.loc[idx, 'AE loss'] = history.history['val_loss'][0]

        # Create plot of losses
        fig, ax = plt.subplots()
        fig.set_size_inches(40, 15)

        SMALL_SIZE = 50
        MEDIUM_SIZE = 60
        BIGGER_SIZE = 70

        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        plt.plot(self._train_metrics['AE loss'], label='Training')
        plt.plot(self._val_metrics['AE loss'], label='Validation')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.grid(True)
        plt.legend()
        #plt.show()
# Initialize model
test_model = Autoencoder(n_features=X_train.shape[1],latent_dim=50, random_seed=1)

# Train the model
test_model.fit(X_train, validation_data=X_test,T_iter=150)

# Standardize the data
scaler = StandardScaler().fit(X)
scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
X = X.pipe(scale_df, scaler)
print("X ", X.shape)
latent_dim = 50

# Generate embeddings
for run in range(10):
    std_model = Autoencoder(n_features=X_train.shape[1],
                            latent_dim=latent_dim, random_seed=run)

    # adverserial train on train set and validate on test set
    std_model.fit(X_train, validation_data=X_test,
                  T_iter=200)

    # Generate embedding for all samples
    embedding = std_model._encoder.predict(X)
    embedding_df = pd.DataFrame(embedding, index=X.index)
    print("Embedding ", embedding_df.shape)
    embedding_df.to_csv('output_path' + 'gene_exp' + str(run) + '.csv')

    # Record models
    model_json = std_model._encoder.to_json()
    with open(output_path+'/AE_encoder_' + str(latent_dim) + 'L_fold' + str(run) + '.json', "w") as json_file:
        json_file.write(model_json)
    std_model._encoder.save_weights(output_path+'/AE_encoder_' + str(latent_dim) + 'L_fold' + str(run) + '.h5')
    print("Saved model to disk")
    model_json = std_model._decoder.to_json()
    with open(output_path+'/AE_encoder_' + str(latent_dim) + 'L_fold' + str(run) + '.json', "w") as json_file:
        json_file.write(model_json)
    std_model._decoder.save_weights(output_path+'/AE_encoder_' + str(latent_dim) + 'L_fold' + str(run) + '.h5')
    print("Saved model to disk")