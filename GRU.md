# Gated Recurrent Unit (GRU)

A Gated Recurrent Unit (GRU) is a type of recurrent neural network (RNN) architecture designed to efficiently capture dependencies in sequential data. Introduced by Kyunghyun Cho et al. in 2014, GRUs address some of the limitations of traditional RNNs, particularly the issues of vanishing and exploding gradients, which hinder the learning of long-term dependencies.

## Core Components of GRU

A GRU cell comprises two primary gates:

### Update Gate ($z_t$)

- **Purpose**: Determines how much of the previous hidden state should be retained.
- **Functionality**: Balances the incorporation of new information with the preservation of existing knowledge.
- **Computation**: 
  \[
  z_t = \sigma(W_z \cdot [x_t, h_{t-1}])
  \]
  Here, $\sigma$ represents the sigmoid activation function, $W_z$ is the weight matrix for the update gate, $x_t$ is the current input, and $h_{t-1}$ is the previous hidden state.

### Reset Gate ($r_t$)

- **Purpose**: Decides how much of the past information to forget.
- **Functionality**: Controls the extent to which the previous hidden state influences the candidate hidden state.
- **Computation**: 
  \[
  r_t = \sigma(W_r \cdot [x_t, h_{t-1}])
  \]
  Similar to the update gate, $W_r$ is the weight matrix for the reset gate.

### Candidate Hidden State ($\tilde{h}_t$)

After determining the gates, the GRU computes a candidate hidden state that represents the new information to be potentially added to the model's memory.

\[
\tilde{h}_t = \tanh(W \cdot [x_t, (r_t * h_{t-1})])
\]

Here, $\tanh$ is the hyperbolic tangent activation function, and the reset gate $r_t$ modulates the influence of the previous hidden state $h_{t-1}$ on the candidate state.

### Final Hidden State ($h_t$)

The final hidden state, which will be passed to the next time step and potentially used for output, is a linear interpolation between the previous hidden state and the candidate hidden state, controlled by the update gate.

\[
h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t
\]

This mechanism allows the GRU to retain information over long sequences selectively, mitigating the vanishing gradient problem and enabling the capture of more extended dependencies compared to traditional RNNs.

## Advantages of GRUs

- **Simplicity**: GRUs have a simpler architecture compared to other gated RNNs like Long Short-Term Memory (LSTM) networks, as they omit the output gate and have fewer parameters.
- **Efficiency**: Due to their streamlined structure, GRUs are computationally efficient and often faster to train.
- **Performance**: GRUs perform comparably to LSTMs on various tasks, making them a popular choice for applications involving sequential data, such as natural language processing, time-series forecasting, and speech recognition.

## Summary

In essence, GRUs enhance the ability of RNNs to model sequential data by incorporating gating mechanisms that control the flow of information. These gates enable the network to decide which information to retain and which to discard, allowing for effective learning of both short-term and long-term dependencies without the complexity inherent in other gated architectures like LSTMs.