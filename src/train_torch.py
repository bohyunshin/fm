import os
import importlib
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

from libs.parse_args import parse_args
from libs.config import load_yaml
from libs.logger import setup_logger
from data.criteo import DataLoader
from preprocess.preprocessor import Preprocessor
from prepare_model_data.torch import prepare_sparse_torch_dataloader

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(ROOT_PATH, "./config/{model}.yaml")
RESULT_PATH = os.path.join(ROOT_PATH, "./result/{test}/{model}/{dt}")


def main(args: ArgumentParser.parse_args):
    # setup result path
    dt = datetime.now().strftime("%Y%m%d%H%M%S")
    test = "test" if args.is_test else "untest"
    result_path = RESULT_PATH.format(test=test, model=args.model, dt=dt)
    os.makedirs(result_path, exist_ok=True)
    logger = setup_logger(os.path.join(result_path, "log.log"))

    config = load_yaml(CONFIG_PATH.format(model=args.model))

    data = DataLoader(
        data_path=args.criteo_data_path,
        logger=logger,
    ).load(is_test=args.is_test)

    # Updated to fit encoder only on training data
    train_data, val_data, test_data, feature_names = Preprocessor(
        categorical_columns=config.data.features.categorical,
        timestamp_column=config.data.extra_columns.timestamp,
        y_column=config.data.extra_columns.label,
    ).fit_and_preprocess(
        data=data,
        val_time_point=datetime.strptime(config.data.split.val_time_point, "%Y-%m-%d"),
        test_time_point=datetime.strptime(
            config.data.split.test_time_point, "%Y-%m-%d"
        ),
    )

    logger.info(f"Number of total data points: {len(data)}")
    logger.info(
        f"Number of data points: train={len(train_data[1])}, val={len(val_data[1])}, test={len(test_data[1])}"
    )
    logger.info(f"Number of features after one-hot encoding: {len(feature_names)}")

    # Prepare sparse dataloaders
    train_dataloader, val_dataloader, test_dataloader = prepare_sparse_torch_dataloader(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
    )

    # set up model
    model_path = f"model.torch.{args.model}"
    model_module = importlib.import_module(model_path).Model
    model = model_module(
        # fm model parameters
        num_features=len(feature_names),
        embedding_dim=args.embedding_dim,
    )
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()
    train_losses = []
    val_losses = []

    # train model
    for epoch in range(args.epochs):
        logger.info(f"################## epoch {epoch} ##################")

        train_loss = 0.0
        val_loss = 0.0

        model.train()
        for features, labels in train_dataloader:
            pred = model(features).squeeze()
            loss = criterion(pred, labels.squeeze())

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss.item())
        logger.info(f"train loss: {round(train_loss.item(), 5)}")

        model.eval()
        with torch.no_grad():
            for features, labels in val_dataloader:
                pred = model(features).squeeze()
                loss = criterion(pred, labels.squeeze())

                val_loss += loss
            val_loss /= len(val_dataloader)
            val_losses.append(val_loss.item())
            logger.info(f"val loss: {round(val_loss.item(), 5)}")

    # report test data metric - now using test_dataloader to avoid oom in dense matrix
    all_predictions = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for features, labels in test_dataloader:
            pred = model(features)
            all_predictions.append(pred.numpy())
            all_labels.append(labels.numpy())

    # Concatenate all predictions and labels
    pred_proba = np.concatenate(all_predictions, axis=0).squeeze()
    y_test = np.concatenate(all_labels, axis=0)
    y_pred = np.array([1 if proba >= 0.5 else 0 for proba in pred_proba])

    auc = roc_auc_score(y_test, pred_proba)
    logger.info(f"AUC-ROC for test data: {round(auc, 5)}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
