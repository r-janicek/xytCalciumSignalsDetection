import argparse

def add_options(parser):
    parser.add_argument("-v","--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-n","--name", default="test",
                        help="specify run name")
    parser.add_argument("-l","--load_epoch", type=int,
                        help="epoch to load if necessary")
    parser.add_argument("-t","--train_epochs", type=int, default=5000,
                        help="number of epochs to run")
    parser.add_argument("-b","--batch_size", type=int, default=1,
                        help="specify batch size")
    parser.add_argument("-s","--small_dataset", action="store_true",
                        help="train using a smaller dataset for testing")
    parser.add_argument("-ss","--very_small_dataset", action="store_true",
                        help="train using a much smaller dataset for testing")
    parser.add_argument("-w0", "--weight_background", type=float, default=1,
                        help="set weight proportion for background")
    parser.add_argument("-w1", "--weight_sparks", type=float, default=1,
                        help="set weight proportion for sparks")
    parser.add_argument("-w2", "--weight_waves", type=float, default=1,
                        help="set weight proportion for waves")
    parser.add_argument("-w3", "--weight_puffs", type=float, default=1,
                        help="set weight proportion for puffs")
