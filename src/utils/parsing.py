def add_arguments(parser):
    # Run parameters
    parser.add_argument("-name", "--run_name", type=str, default="local", help="Run id")
    parser.add_argument("-m", "--augmentation", type=int, default=None,
                        help="Factor to increase number of images with through augmentation", )
    parser.add_argument("-train", "--train", type=bool, default=True, help="Train model?")
    parser.add_argument("-test", "--test", type=bool, default=False, help="Test model?")
    parser.add_argument("-model", "--model", type=str, default=None, help="Existing model. If None, train new one.", )
    parser.add_argument("-fold", "--fold", type=int, default=0, help="Choose a fold. Usually integer between 0 and 5. ")

    # Training parameters
    parser.add_argument("-vae", "--variational_ae", type=bool, default=True,
                        help="Choose Convolutional Variational Autoencoder", )
    parser.add_argument("-t", "--architecture", type=str, default="expanding",
                        help="Network architecture. Either 'fully', 'contracting' or 'expanding'. ", )
    parser.add_argument("-d", "--depth", type=int, default=4, help="Number of Conv2D layers until bottleneck.", )
    parser.add_argument("-dense", "--dense", type=int, default=None, help="Number of Dense layers at bottleneck.", )
    parser.add_argument("-f", "--filters", type=int, default=4, help="Number of initial conv-filters in autoencoder", )
    parser.add_argument("-l", "--loss", type=str, default="mse_spectral", help="Loss function")
    parser.add_argument("-e", "--epochs", type=int, default=30, help="Epochs")
    parser.add_argument("-s", "--steps", type=int, default=8, help="Steps per epoch")
    parser.add_argument("-b", "--batch_size", type=int, default=8,
                        help="Training batch size for image generator. Don't choose too high (>8) if you don't want to risk the system freezing.", )
    # parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="Learning rate for training optimizer")
    parser.add_argument("-beta", "--beta", type=float, default=100.0, help="Ratio between spectral loss and mse", )
    parser.add_argument("-k", "--ksize", type=int, default=None, help="Kernel size for average filter")
    parser.add_argument("-p", "--patience", type=int, default=None, help="Early stopping patience")

    args = parser.parse_args()
    return args
