import keras.backend as K
from keras.activations import tanh, softmax
from keras.engine import InputSpec
from keras.layers import LSTM
import keras


class Attention(keras.layers.Layer):
    """
        Attention layer
    """

    def __init__(self, hidden_dimensions, name='attention'):
        super(Attention, self).__init__(name=name, trainable=True)
        self.W1 = keras.layers.Dense(hidden_dimensions, use_bias=False)
        self.W2 = keras.layers.Dense(hidden_dimensions, use_bias=False)
        self.V = keras.layers.Dense(1, use_bias=False)

    def call(self, encoder_outputs, dec_output, mask=None):

        w1_e = self.W1(encoder_outputs)
        w2_d = self.W2(dec_output)
        tanh_output = tanh(w1_e + w2_d)
        v_dot_tanh = self.V(tanh_output)
        if mask is not None:
            v_dot_tanh += (mask * -1e9)
        attention_weights = softmax(v_dot_tanh, axis=1)
        att_shape = K.shape(attention_weights)
        return K.reshape(attention_weights, (att_shape[0], att_shape[1]))


class Decoder(keras.layers.Layer):
    """
        Decoder class for PointerLayer
    """

    def __init__(self, hidden_dimensions):
        super(Decoder, self).__init__()
        self.lstm = keras.layers.LSTM(
            hidden_dimensions, return_sequences=False, return_state=True)

    def call(self, x, hidden_states):
        dec_output, state_h, state_c = self.lstm(
            x, initial_state=hidden_states)
        return dec_output, [state_h, state_c]

    def get_initial_state(self, inputs):
        return self.lstm.get_initial_state(inputs)

    def process_inputs(self, x_input, initial_states, constants):
        return self.lstm._process_inputs(x_input, initial_states, constants)


class PointerLSTM(keras.layers.Layer):
    """
        PointerLSTM
    """

    def __init__(self, hidden_dimensions, name='pointer', **kwargs):
        super(PointerLSTM, self).__init__(
            hidden_dimensions, name=name, **kwargs)
        self.hidden_dimensions = hidden_dimensions
        self.attention = Attention(hidden_dimensions)
        self.decoder = Decoder(hidden_dimensions)

    def build(self, input_shape):
        super(PointerLSTM, self).build(input_shape)
        self.input_spec = [InputSpec(shape=input_shape)]

    def call(self, x, training=None, mask=None, states=None):
        """
        :param Tensor x: Should be the output of the decoder
        :param Tensor states: last state of the decoder
        :param Tensor mask: The mask to apply
        :return: Pointers probabilities
        """

        input_shape = self.input_spec[0].shape
        en_seq = x
        x_input = x[:, input_shape[1] - 1, :]
        x_input = K.repeat(x_input, input_shape[1])
        if states:
            initial_states = states
        else:
            initial_states = self.decoder.get_initial_state(x_input)

        constants = []
        preprocessed_input, _, constants = self.decoder.process_inputs(
            x_input, initial_states, constants)
        constants.append(en_seq)
        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.decoder.lstm.go_backwards,
                                             constants=constants,
                                             input_length=input_shape[1])

        return outputs

    def step(self, x_input, states):
        x_input = K.expand_dims(x_input,1)
        input_shape = self.input_spec[0].shape
        en_seq = states[-1]
        _, [h, c] = self.decoder(x_input, states[:-1])
        dec_seq = K.repeat(h, input_shape[1])
        probs = self.attention(dec_seq, en_seq)
        return probs, [h, c]

    def get_output_shape_for(self, input_shape):
        # output shape is not affected by the attention component
        return (input_shape[0], input_shape[1], input_shape[1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1])
