using Tensorflow;
using static Tensorflow.KerasApi;
namespace SharpGippetty._TransformerModel
{
   
    public class DecoderLayer
    {

        private readonly MainWindow _mainwindow;
        public DecoderLayer(MainWindow mainWindow)
        {
            _mainwindow = mainWindow;

        }
        public Tensor decoderLayer(Tensor input, Tensor lookAheadMask, Tensor paddingMask)
        {
            var self_attention = _mainwindow._selfAttentionLayer.SelfAttention(_mainwindow._transformer.d_model, _mainwindow._transformer.num_heads)(input, lookAheadMask, paddingMask);
            var attention_output = keras.layers.LayerNormalization(axis: -1, epsilon: 1e-6f).Apply(input + self_attention);
            var dropout = keras.layers.Dropout(_mainwindow._transformer.dropoutrate);
            attention_output = dropout.Apply(attention_output);
            var feed_forward = _mainwindow._feedForwardNetwork.FeedForward(_mainwindow._transformer.d_model, _mainwindow._transformer.dff)(attention_output);
            var output = keras.layers.LayerNormalization(axis: -1, epsilon: 1e-6f).Apply(attention_output + feed_forward);
            return output;
        }
    }
}
