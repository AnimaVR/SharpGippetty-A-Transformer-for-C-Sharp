using Tensorflow;
using static Tensorflow.KerasApi;
using System;

namespace SharpGippetty._TransformerModel
{
  
    public class FeedForwardNetwork
    {

        private readonly MainWindow _mainWindow;
        public FeedForwardNetwork(MainWindow mainWindow)
        {
            _mainWindow = mainWindow;
           
        }
       public Func<Tensor, Tensor> FeedForward(int d_model, int dff)
        {
            var dense1 = keras.layers.Dense(dff, activation: "relu");
            var dense2 = keras.layers.Dense(d_model);
            return (input) =>
            {
                var output = dense1.Apply(input);
                output = dense2.Apply(output);

                return output;
            };
        }
    }
}
