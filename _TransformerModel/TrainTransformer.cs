using Tensorflow;
using static Tensorflow.KerasApi;
using static Tensorflow.Binding;
using System.IO;
using System;
using System.Collections.Generic;
using System.Linq;
namespace SharpGippetty._TransformerModel
{
   
    public class TrainTransformer
    {
       
        private readonly MainWindow _mainwindow;
        public TrainTransformer(MainWindow mainWindow)
        {
            _mainwindow = mainWindow;
            
        }

        public void Train(string trainDataPath, string validationDataPath)
        {
            _mainwindow.Dispatcher.Invoke(() => _mainwindow.ResultBox.AppendText("Training started.\n"));

            var trainDataChunks = LoadData(trainDataPath);
            var validationDataChunks = LoadData(validationDataPath);
            int epochs = 5;
            int batch_size = _mainwindow._transformer.batchSize; // Use the batch size from the transformer instance.

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                float currentLearningRate = CustomScheduleLearningRateDecay(epoch + 1);
                var optimizer = keras.optimizers.Adam(learning_rate: currentLearningRate);

                // Iterate over the training data chunks
                for (int chunkIdx = 0; chunkIdx < trainDataChunks.Count; chunkIdx++)
                {
                    var trainData = trainDataChunks[chunkIdx];
                    var trainX = tf.slice(trainData, new int[] { 0, 0 }, new int[] { (int)trainData.shape[0] - 1, (int)trainData.shape[1] });
                    var trainY = tf.slice(trainData, new int[] { 1, 0 }, new int[] { (int)trainData.shape[0] - 1, (int)trainData.shape[1] });

                    for (int i = 0; i < trainX.shape[0]; i += batch_size)
                    {
                        var batchX = trainX.slice(new Slice(i, i + batch_size));
                        var batchY = trainY.slice(new Slice(i, i + batch_size));
                        using (var tape = tf.GradientTape())
                        {
                            var predictions = _mainwindow._transformer._model.Apply(batchX, training: true);
                            var mask = _mainwindow._selfAttentionLayer.CreatePaddingMask(batchX);
                            var selfAttentionOutput = _mainwindow._selfAttentionLayer.SelfAttention(_mainwindow._transformer.d_model, _mainwindow._transformer.num_heads)(batchX, mask, mask);

                            var decoderOutput = _mainwindow._decoderLayer.decoderLayer(selfAttentionOutput, selfAttentionOutput, mask);
                            var lossValue = keras.losses.SparseCategoricalCrossentropy(from_logits: true).Call(batchY, decoderOutput);
                            var scaledLossValue = lossValue / batch_size;
                            var gradients = tape.gradient(scaledLossValue, tf.trainable_variables());
                            ApplyGradients(gradients.Zip(tf.trainable_variables(), (gradient, variable) => (gradient, variable)));
                        }
                        _mainwindow.Dispatcher.Invoke(() => _mainwindow.ResultBox.AppendText($"Chunk {chunkIdx + 1} completed.\n"));
                    }
                    string checkpointPath = $"model_checkpoint_epoch{epoch + 1}.h5";
                    _mainwindow._transformer.SaveWeights(checkpointPath);
                    _mainwindow.Dispatcher.Invoke(() => _mainwindow.ResultBox.AppendText($"Epoch {epoch + 1} completed. Model weights saved at {checkpointPath}\n"));

                    // Similarly, iterate over the validation data chunks for evaluation
                    foreach (var valData in validationDataChunks)
                    {
                        var valX = tf.slice(valData, new int[] { 0, 0 }, new int[] { (int)valData.shape[0] - 1, (int)valData.shape[1] });
                        var valY = tf.slice(valData, new int[] { 1, 0 }, new int[] { (int)valData.shape[0] - 1, (int)valData.shape[1] });
                        var results = _mainwindow._transformer._model.evaluate(valX.numpy(), valY.numpy());
                    }
                }
                _mainwindow.Dispatcher.Invoke(() => _mainwindow.ResultBox.AppendText("Training completed.\n"));
            }
        }

        private void ApplyGradients(IEnumerable<(Tensor gradient, IVariableV1 variable)> gradientsAndVariables)
        {
            var optimizer = keras.optimizers.Adam();
            var gradsAndVars = gradientsAndVariables.Select(tuple => (tuple.gradient, tuple.variable));
            optimizer.apply_gradients(gradsAndVars);
        }

        public float CustomScheduleLearningRateDecay(int step)
        {
            float warmup_steps = 200.0f;
            float d_model = _mainwindow._transformer.d_model; 
            float arg1 = 1 / MathF.Sqrt(step);
            float arg2 = step * MathF.Pow(warmup_steps, -1.5f);
            return (1 / MathF.Sqrt(d_model)) * MathF.Min(arg1, arg2);
        }

        public List<Tensor> LoadData(string path, int chunkSize = 2048)
        {
            List<Tensor> dataChunks = new List<Tensor>();
            using (BinaryReader reader = new BinaryReader(File.OpenRead(path)))
            {
                while (reader.BaseStream.Position != reader.BaseStream.Length)
                {
                    List<int> chunkData = new List<int>();
                    for (int i = 0; i < chunkSize; i++)
                    {
                        if (reader.BaseStream.Position != reader.BaseStream.Length)
                        {
                            chunkData.Add(reader.ReadInt32());
                        }
                        else
                        {
                            break;
                        }
                    }

                    dataChunks.Add(tf.reshape(tf.constant(chunkData.ToArray()), new int[] { -1, chunkSize }));
                }
            }
            return dataChunks;
        }


    }
}
