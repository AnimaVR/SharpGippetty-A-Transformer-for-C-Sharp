using Tensorflow;
using static Tensorflow.KerasApi;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace SharpGippetty._TransformerModel
{
    public class TransformerModel
    {
        public int num_heads { get; private set; }
        public int d_model { get; private set; }
        public int dff { get; private set; }
        public int num_layers { get; private set; }
        public float dropoutrate { get; private set; }
        public int maxTokens { get; private set; }
        public int vocab_size { get; private set; }
        public int batchSize { get; private set; }
        public MainWindow _mainwindow;
        public IModel _model;

        public TransformerModel(MainWindow mainWindow, int numLayers = 4, int numHeads = 4, int dModel = 4096, int dff = 1028, float dropout_rate = 0.1f, int batchSize = 2048)
        {
            _mainwindow = mainWindow;
            maxTokens = batchSize;
            num_heads = numHeads;
            d_model = dModel;
            this.dff = dff;
            num_layers = numLayers;
            dropoutrate = dropout_rate;
            vocab_size = 71;
            this.batchSize = batchSize;
        }
        public void InitializeAndTrain(string trainDataPath, string validationDataPath)
        {
            // First, we build the model
            Model(maxTokens);

            // Then, we initialize HelperFunctions class and train the model
            _mainwindow._train.Train(trainDataPath, validationDataPath);
        }

        public void Model(int maxTokens)
        {
            var config = new ConfigProto
            {
                GpuOptions = new GPUOptions
                {
                    AllowGrowth = true,
                    PerProcessGpuMemoryFraction = 0.9 // Adjust the memory fraction based on your GPU's memory capacity
                }
            };
            using (var session = tf.compat.v1.Session())
            {
                session.run(tf.global_variables_initializer());
                var input = keras.Input(shape: new Tensorflow.Shape(maxTokens));
                var embedded_input = _mainwindow._embeddingLayer.embeddingLayer(input); // Use the instance
                var encoded_input = _mainwindow._selfAttentionLayer.AddPositionalEncoding(embedded_input, maxTokens);
                var dropout = keras.layers.Dropout(dropoutrate);
                var x = dropout.Apply(encoded_input);
                var lookAheadMask = _mainwindow._selfAttentionLayer.CreateLookAheadMask(maxTokens);
                var paddingMask = _mainwindow._selfAttentionLayer.CreatePaddingMask(input);
                for (int i = 0; i < num_layers; i++)
                {
                    x = _mainwindow._selfAttentionLayer.SelfAttention(d_model, num_heads)(x, paddingMask, lookAheadMask);
                }

                x = keras.layers.Dense(d_model, activation: "relu").Apply(x);
                var output = keras.layers.Dense(vocab_size, activation: null, use_bias: true).Apply(x); // Identity activation
                var model = keras.Model(input, output);
                model.compile(optimizer: keras.optimizers.Adam(learning_rate: 0.001f), loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true), metrics: new[] { "accuracy" });
                _model = model;
            }
        }
        public void SaveWeights(string path)
        {
            _model.save_weights(path);
        }
        public void LoadWeights(string path)
        {
            _model.load_weights(path);
        }
    }
}
