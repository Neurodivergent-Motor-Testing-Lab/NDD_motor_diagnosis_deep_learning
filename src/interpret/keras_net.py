from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
import tensorflow as tf
import torch

# Optionally, set the LSTM layer to use CPU instead of GPU
tf.config.set_visible_devices([], 'GPU')
def build_keras_net(input_dim, hidden_dim, output_dim, num_lstm_layers=1, dropout=0.3):
    # Input shape: (seq_len, batch_size, input_dim) in PyTorch
    # In Keras, we typically use: (batch_size, seq_len, input_dim)
    # So we set input shape as (None, input_dim) assuming variable-length sequences
    # inputs = Input(shape=(None, input_dim))  # (batch_size, seq_len, input_dim)
    inputs = Input(shape=(40, input_dim))  # (batch_size, seq_len, input_dim)

    x = inputs

    for i in range(num_lstm_layers):
        return_sequences = i < num_lstm_layers - 1
        x = LSTM(hidden_dim, return_sequences=return_sequences, dropout=dropout)(x)

    x = Dropout(dropout)(x)  # Don't overwrite the dropout variable!
    outputs = Dense(output_dim, kernel_initializer='random_normal')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def transfer_weights(pytorch_path, keras_model, device, num_lstm_layers=1):
    # Load PyTorch weights
    pt_weights = torch.load(pytorch_path, map_location=device, weights_only=False)
    pt_weights = pt_weights['model_state_dict']

    # === LSTM weights ===
    for i in range(num_lstm_layers):
        # PyTorch names
        w_ih = tf.convert_to_tensor(pt_weights[f'lstm.weight_ih_l{i}'].cpu().numpy().T ) # (4*hidden, input)
        w_hh = tf.convert_to_tensor(pt_weights[f'lstm.weight_hh_l{i}'].cpu().numpy().T)  # (4*hidden, hidden)
        b_ih = tf.convert_to_tensor(pt_weights[f'lstm.bias_ih_l{i}'].cpu().numpy())
        b_hh = tf.convert_to_tensor(pt_weights[f'lstm.bias_hh_l{i}'].cpu().numpy())
        # w_ih = pt_weights[f'lstm.weight_ih_l{i}'].T  # (4*hidden, input)
        # w_hh = pt_weights[f'lstm.weight_hh_l{i}'].T  # (4*hidden, hidden)
        # b_ih = pt_weights[f'lstm.bias_ih_l{i}']
        # b_hh = pt_weights[f'lstm.bias_hh_l{i}']
        bias = b_ih + b_hh  # Keras uses one bias

        # Get Keras LSTM layer
        lstm_layer = None
        lstm_count = 0
        for layer in keras_model.layers:
            if isinstance(layer, tf.keras.layers.LSTM):
                if lstm_count == i:
                    lstm_layer = layer
                    break
                lstm_count += 1

        if lstm_layer is None:
            raise ValueError(f"Couldn't find LSTM layer {i} in the Keras model.")

        keras_weights = [w_ih, w_hh, bias]
        lstm_layer.set_weights(keras_weights)

    # === Linear (Dense) layer ===
    linear_w = tf.convert_to_tensor(pt_weights['linear.weight'].cpu().numpy().T)  # PyTorch: [out, in], Keras: [in, out]
    linear_b = tf.convert_to_tensor(pt_weights['linear.bias'].cpu().numpy())

    # linear_w = pt_weights['linear.weight'].T  # PyTorch: [out, in], Keras: [in, out]
    # linear_b = pt_weights['linear.bias']

    # Find final Dense layer
    for layer in reversed(keras_model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            layer.set_weights([linear_w, linear_b])
            break
