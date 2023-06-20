using SharpGippetty._TransformerModel;
using Tensorflow;
using static Tensorflow.KerasApi;
using static Tensorflow.Binding;
using System;

namespace SharpGippetty._TransformerModel
{
   
    public class SelfAttentionLayer
    {
 
        private readonly MainWindow _mainwindow;
        public SelfAttentionLayer(MainWindow mainWindow)
        {
            _mainwindow = mainWindow;
   
        }
        public Tensor CreatePaddingMask(Tensor seq)
        {
            var zeroTensor = tf.constant(0, dtype: seq.dtype);
            var mask = tf.cast(tf.equal(seq, zeroTensor), tf.float32);
            return tf.expand_dims(tf.expand_dims(mask, axis: 1), axis: 1);
        }


        public Tensor CreateLookAheadMask(int size)
        {
            var mask = tf.ones((size, size));
            var indices = new long[size * (size - 1) / 2][];
            var values = new float[size * (size - 1) / 2];
            var count = 0;
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    indices[count] = new long[] { i, j };
                    values[count] = 0;
                    count++;
                }
            }
            var indicesTensor = tf.constant(indices, dtype: tf.int64);
            var valuesTensor = tf.constant(values, dtype: tf.float32);
            var scatterMask = ScatterND(indicesTensor, valuesTensor, tf.constant(new long[] { size, size }));
            mask = tf.multiply(mask, scatterMask);
            return mask;
        }


        public Tensor ScatterND(Tensor indices, Tensor values, Tensor shape)
        {
            var scatterMask = tf.zeros(shape, dtype: tf.float32);
            var sess = tf.compat.v1.Session();
            var scatterMaskArray = sess.run(scatterMask);
            var indicesArray = indices.numpy().ToArray<int>();
            var valuesArray = values.numpy().ToArray<float>();
            for (int i = 0; i < indicesArray.Length; i++)
            {
                var index = indicesArray[i];
                var value = valuesArray[i];

                scatterMaskArray[index] = value;
            }
            return tf.constant(scatterMaskArray);
        }

        public Tensor AddPositionalEncoding(Tensor input, int seqLength)
        {
            var seq_length = tf.shape(input)[1];
            var position = tf.range(tf.constant(0, dtype: tf.int32), seq_length, tf.constant(1, dtype: tf.int32));
            var div_term = tf.pow(tf.constant(10000, dtype: tf.float32), tf.range(tf.constant(0), tf.constant(_mainwindow._transformer.d_model), tf.constant(2), dtype: tf.float32) / tf.cast(tf.constant(_mainwindow._transformer.d_model), tf.float32));

            var encodings = tf.cast(position, dtype: tf.float32) * tf.expand_dims(div_term, axis: 0);
            var sin_encodings = tf.sin(encodings);
            var cos_encodings = tf.cos(encodings);
            var pos_encodings = tf.concat(new[] { sin_encodings, cos_encodings }, axis: -1);

            var inputShape = tf.shape(input);
            var batchShape = tf.slice(inputShape, new[] { 0 }, new[] { 1 });
            var pos_encodingsBroadcasted = tf.tile(pos_encodings, new[] { batchShape[0], tf.constant(seq_length), tf.constant(_mainwindow._transformer.d_model) });

            var masked_input = input * tf.cast(tf.constant(0, dtype: input.dtype), dtype: input.dtype);
            var input_with_encoding = tf.add(masked_input, pos_encodingsBroadcasted);

            return input_with_encoding;
        }

        public Func<Tensor, Tensor> SplitHeads(int num_heads, int depth)
        {
            return (input) =>
            {
                int batch_size = (int)input.shape[0];
                var split_input = tf.reshape(input, new int[] { batch_size, -1, num_heads, depth });
                return tf.transpose(split_input, new int[] { 0, 2, 1, 3 });
            };
        }

        public Func<Tensor, Tensor, Tensor, Tensor> SelfAttention(int d_model, int num_heads)
        {
            var num_units = d_model / num_heads;
            var depth = d_model / num_heads;
            var dense_query = keras.layers.Dense(d_model);
            var dense_key = keras.layers.Dense(d_model);
            var dense_value = keras.layers.Dense(d_model);
            var split_heads = SplitHeads(num_heads, depth);
            var concat_heads = keras.layers.Concatenate(axis: -1);
            var dense_output = keras.layers.Dense(d_model);
            var selfAttentionLayer = this;

            return (input, mask, paddingMask) =>
            {
                var query = dense_query.Apply(input);
                var key = dense_key.Apply(input);
                var value = dense_value.Apply(input);

                // Apply the masks to the scores
                var scores = tf.matmul(query, tf.transpose(key, new int[] { 0, 2, 1 }));
                scores += paddingMask * tf.constant(float.MinValue);

                // Apply the masking for future positions
                scores += mask * tf.constant(float.MinValue);

                var scaled_scores = scores / tf.sqrt(tf.cast(tf.constant(num_units), tf.float32));
                var attention_weights = tf.nn.softmax(scaled_scores, axis: -1);
                var attention_output = tf.matmul(attention_weights, value);
                attention_output = concat_heads.Apply(tf.transpose(attention_output, new int[] { 0, 2, 1, 3 }));
                return dense_output.Apply(attention_output);
            };
        }


    }
}
