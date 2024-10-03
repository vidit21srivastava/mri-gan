import argparse
from pipeline import MRIGANPipeline


def parse_args():

    parser = argparse.ArgumentParser(description="MRI GAN Training Pipeline")

    parser.add_argument('--data_dir_t1', type=str, default="./Tr1/",
                        help="Path to the T1 MRI dataset directory")
    parser.add_argument('--data_dir_t2', type=str, default="./Tr2/",
                        help="Path to the T2 MRI dataset directory")
    parser.add_argument('--img_height', type=int, default=256,
                        help="Height of the input images")
    parser.add_argument('--img_width', type=int, default=256,
                        help="Width of the input images")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for training")
    parser.add_argument('--buffer_size', type=int, default=1000,
                        help="Buffer size for shuffling the dataset")
    parser.add_argument('--epochs', type=int, default=125,
                        help="Number of epochs to train the model")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    pipeline = MRIGANPipeline(
        data_dir_t1=args.data_dir_t1,
        data_dir_t2=args.data_dir_t2,
        img_height=args.img_height,
        img_width=args.img_width,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        epochs=args.epochs
    )

    pipeline.train()
