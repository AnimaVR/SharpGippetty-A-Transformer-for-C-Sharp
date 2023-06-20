using SharpGippetty._TransformerModel;
using Tensorflow;
using static Tensorflow.KerasApi;
using static Tensorflow.Binding;
using System;

namespace SharpGippetty._TransformerModel
{
    
    public class EncoderLayer
    {
        Transformer _transformer;
        private readonly MainWindow _mainWindow;
        public EncoderLayer(MainWindow mainWindow, Transformer transformer)
        {
            _mainWindow = mainWindow;
            _transformer = transformer;
        }
        private Tensor encoderLayer(Tensor input, Tensor paddingMask)
        {
            var attention = _transformer._selfattentionLayer.BuildSelfAttentionLayer(_transformer.d_model, _transformer.num_heads)(input, paddingMask, paddingMask);

            var attention_output = keras.layers.LayerNormalization(axis: -1, epsilon: 1e-6f).Apply(input + attention);
           var dropout = keras.layers.Dropout(_transformer.dropoutrate);
           attention_output = dropout.Apply(attention_output);
           var feed_forward =_transformer._feedForwardNetwork.feedForwardNetwork(_transformer.d_model, _transformer.dff)(attention_output);
           var output = keras.layers.LayerNormalization(axis: -1, epsilon: 1e-6f).Apply(attention_output + feed_forward);
           return output;
        }
        private Func<Tensor, Tensor, Tensor, Tensor> EncoderAttentionLayer(int d_model, int num_heads)
        {
            var num_units = d_model / num_heads;
            var depth = d_model / num_heads;
            var dense_query = keras.layers.Dense(d_model);
            var dense_key = keras.layers.Dense(d_model);
            var dense_value = keras.layers.Dense(d_model);
            var split_heads = _transformer._selfattentionLayer.SplitHeads(num_heads, depth);
            var concat_heads = keras.layers.Concatenate(axis: -1);
            var dense_output = keras.layers.Dense(d_model);
            return (input_decoder, input_encoder, mask) =>
            {
                var query = dense_query.Apply(input_decoder);
                var key = dense_key.Apply(input_encoder);
                var value = dense_value.Apply(input_encoder);
                var scores = tf.matmul(query, tf.transpose(key, new int[] { 0, 2, 1 }));
                var scaled_scores = scores / tf.sqrt(tf.cast(tf.constant(num_units), tf.float32));
                var attention_weights = tf.nn.softmax(scaled_scores + mask * tf.constant(float.MinValue), axis: -1);
                var attention_output = tf.matmul(attention_weights, value);
                attention_output = concat_heads.Apply(tf.transpose(attention_output, new int[] { 0, 2, 1, 3 }));
                return dense_output.Apply(attention_output);
            };
        }
    }
}
