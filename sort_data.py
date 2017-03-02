from __future__ import absolute_import, division, print_function
import numpy as np


class DataGenerator(object):
    def next_batch(self, batch_size, N, train_mode=True):
        """Return the next `batch_size` examples from this data set."""

        # A sequence of random numbers from [0, 1]
        encoder_batch = []

        # Sorted sequence that we feed to encoder
        # In inference we feed an unordered sequence again
        decoder_batch = []

        # Ordered sequence where one hot vector encodes
        # position in the input array
        target_batch = []
        for _ in range(batch_size):
            encoder_batch.append(np.zeros([N, 1]))
        for _ in range(batch_size):
            decoder_batch.append(np.zeros([N, 1]))
            target_batch.append(np.zeros([N, N]))

        encoder_batch = np.asarray(encoder_batch)
        decoder_batch = np.asarray(decoder_batch)
        target_batch = np.asarray(target_batch)

        for b in range(batch_size):
            shuffle = np.random.permutation(N)
            sequence = np.sort(np.random.random(N))
            shuffled_sequence = sequence[shuffle]

            for i in range(N):
                encoder_batch[b][i] = shuffled_sequence[i]
                if train_mode:
                    decoder_batch[b][i] = sequence[i]
                else:
                    decoder_batch[b][i] = shuffled_sequence[i]
                target_batch[b, i][shuffle[i]] = 1.0

            # Points to the stop symbol
            # target_batch[b, N][0] = 1.0

        return encoder_batch, decoder_batch, target_batch


if __name__ == "__main__":
    seq_len = 3
    batch_size = 3
    dataset = DataGenerator()
    enc_input, dec_input, targets = dataset.next_batch(batch_size, seq_len)
    print("batch_size", batch_size, "seq_len", seq_len)
    print("-------------encoder input-------------")
    print(enc_input.shape)
    print(enc_input)
    print("-------------decoder input-------------")
    print(dec_input.shape)
    print(dec_input)
    print("-------------   targets   -------------")
    print(targets.shape)
    print(targets)
