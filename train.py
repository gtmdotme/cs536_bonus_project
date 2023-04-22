import click
from pytorch_lightning.loggers import TensorBoardLogger

from ml.utils import (
    train_mlp,
    train_cnn,
    train_sae,
    train_resnet
)

# hyperparameters
INPUT_DIM = 1500
EPOCHS = 10
OUTPUT_DIM_APPLICATION = 17
OUTPUT_DIM_TRAFFIC = 12

@click.command()
@click.option(
    "-d",
    "--data_path",
    help="training data dir path containing parquet files.",
    required=True,
)
@click.option(
    "-m", 
    "--model_path", 
    help="output model path", 
    required=True)
@click.option(
    "-t",
    "--task",
    help='classification task. Option: "app" or "traffic".',
    required=True,
)
@click.option(
    "-a",
    "--architecture",
    help='architecture (model) to be used for classification. \
            Option: "cnn" or "mlp" or "sae" or "resnet".',
    required=True,
)

def main(data_path, model_path, task, architecture):
    if architecture == "mlp":
        if task == "app":
            # train_application_classification_mlp_model(data_path, model_path)
            logger = TensorBoardLogger(
                "application_classification_mlp_logs", "application_classification_mlp"
            )
            train_mlp(
                output_dim=OUTPUT_DIM_APPLICATION,
                data_path=data_path,
                epoch=EPOCHS,
                model_path=model_path,
                signal_length=INPUT_DIM,
                logger=logger,
            )
        elif task == "traffic":
            # train_traffic_classification_mlp_model(data_path, model_path)
            logger = TensorBoardLogger(
                "traffic_classification_mlp_logs", "traffic_classification_mlp"
            )
            train_mlp(
                output_dim=OUTPUT_DIM_TRAFFIC,
                data_path=data_path,
                epoch=EPOCHS,
                model_path=model_path,
                signal_length=INPUT_DIM,
                logger=logger,
            )
        else:
            exit("Not Support")
    elif architecture == "cnn":
        if task == "app":
            # train_application_classification_cnn_model(data_path, model_path)
            logger = TensorBoardLogger(
                "application_classification_cnn_logs", "application_classification_cnn"
            )
            train_cnn(
                c1_kernel_size=4,
                c1_output_dim=200,
                c1_stride=3,
                c2_kernel_size=5,
                c2_output_dim=200,
                c2_stride=1,
                output_dim=OUTPUT_DIM_APPLICATION,
                data_path=data_path,
                epoch=EPOCHS,
                model_path=model_path,
                signal_length=INPUT_DIM,
                logger=logger,
            )
        elif task == "traffic":
            # train_traffic_classification_cnn_model(data_path, model_path)
            logger = TensorBoardLogger(
                "traffic_classification_cnn_logs", "traffic_classification_cnn"
            )
            train_cnn(
                c1_kernel_size=5,
                c1_output_dim=200,
                c1_stride=3,
                c2_kernel_size=4,
                c2_output_dim=200,
                c2_stride=3,
                output_dim=OUTPUT_DIM_TRAFFIC,
                data_path=data_path,
                epoch=EPOCHS,
                model_path=model_path,
                signal_length=INPUT_DIM,
                logger=logger,
            )
        else:
            exit("Not Support")
    elif architecture == "sae":
        if task == "app":
            # train_application_classification_sae_model(data_path, model_path)
            logger = TensorBoardLogger(
                "application_classification_sae_logs", "application_classification_sae"
            )
            train_sae(
                output_dim=OUTPUT_DIM_APPLICATION,
                data_path=data_path,
                epoch=EPOCHS,
                model_path=model_path,
                signal_length=INPUT_DIM,
                logger=logger
            )
        elif task == "traffic":
            # train_traffic_classification_sae_model(data_path, model_path)
            logger = TensorBoardLogger(
                "traffic_classification_sae_logs", "traffic_classification_sae"
            )
            train_sae(
                output_dim=OUTPUT_DIM_TRAFFIC,
                data_path=data_path,
                epoch=EPOCHS,
                model_path=model_path,
                signal_length=INPUT_DIM,
                logger=logger
            )
        else:
            exit("Not Support")
    elif architecture == "resnet":
        if task == "app":
            # train_application_classification_resnet_model(data_path, model_path)
            logger = TensorBoardLogger(
                "application_classification_resnet_logs", "application_classification_resnet"
            )
            train_resnet(
                c1_kernel_size=4,
                c1_output_dim=16,
                c1_stride=3,
                c1_groups=1,
                c1_n_block=4,
                output_dim=OUTPUT_DIM_APPLICATION,
                data_path=data_path,
                epoch=EPOCHS,
                model_path=model_path,
                signal_length=INPUT_DIM,
                logger=logger,
            )
        elif task == "traffic":
            # train_traffic_classification_resnet_model(data_path, model_path)
            logger = TensorBoardLogger(
                "traffic_classification_resnet_logs", "traffic_classification_resnet"
            )
            train_resnet(
                c1_kernel_size=5,
                c1_output_dim=16,
                c1_stride=3,
                c1_groups=1,
                c1_n_block=4,
                output_dim=OUTPUT_DIM_TRAFFIC,
                data_path=data_path,
                epoch=EPOCHS,
                model_path=model_path,
                signal_length=INPUT_DIM,
                logger=logger,
            )
        else:
            exit("Not Support")
    else:
        exit("Not Support")


if __name__ == "__main__":
    main()
