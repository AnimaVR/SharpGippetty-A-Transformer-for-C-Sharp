using SharpGippetty._TransformerModel;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace SharpGippetty
{
    
    public partial class MainWindow : Window
    {
        public EmbeddingLayer _embeddingLayer;
        public DecoderLayer _decoderLayer;
        public SelfAttentionLayer _selfAttentionLayer;
        public FeedForwardNetwork _feedForwardNetwork;
        public TrainTransformer _train;
        public TransformerModel _transformer;
        public MainWindow()
        {
            InitializeComponent();
            _transformer = new TransformerModel(this);
            _embeddingLayer = new EmbeddingLayer(this);
            _decoderLayer = new DecoderLayer(this);
            _selfAttentionLayer = new SelfAttentionLayer(this);
            _feedForwardNetwork = new FeedForwardNetwork(this);
            _train = new TrainTransformer(this);
        }
        private void BuildLLMButton_Click(object sender, RoutedEventArgs e)
        {

            _transformer.InitializeAndTrain("train.bin", "val.bin");

        }
    }
}
