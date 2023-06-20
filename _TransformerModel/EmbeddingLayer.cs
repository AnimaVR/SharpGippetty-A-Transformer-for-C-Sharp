using Tensorflow;
using static Tensorflow.KerasApi;
using static Tensorflow.Binding;
namespace SharpGippetty._TransformerModel
{
    public class EmbeddingLayer
    {
        private readonly MainWindow _mainwindow;
        public Tensorflow.Keras.Layers.Embedding embedding;

        public EmbeddingLayer(MainWindow mainWindow)
        {
            _mainwindow = mainWindow;
            var vocab_size = _mainwindow._transformer.vocab_size;
            var embedding_dim = _mainwindow._transformer.d_model;
            this.embedding = keras.layers.Embedding(vocab_size, embedding_dim) as Tensorflow.Keras.Layers.Embedding;
        }

        public Tensor embeddingLayer(Tensor input)
        {
            var embeddings = this.embedding.Apply(input);
            var padding_mask = _mainwindow._selfAttentionLayer.CreatePaddingMask(input);
            var reshaped_padding_mask = tf.expand_dims(padding_mask, axis: -1);
            var masked_embeddings = embeddings * reshaped_padding_mask;
            return masked_embeddings;
        }
    }
}
