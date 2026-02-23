import deeplabcut as dlc
import os
import shutil
import zipfile
import logging
from typing import Optional, List, Union
import json

from ideas.exceptions import IdeasError
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
import seaborn as sns
import cv2

from toolbox.utils.utilities import (
    get_timing_spacing_metadata_from_labeled_movie,
)

import yaml

# constants
GPU_TO_USE = 0
DLC_METADATA_KEY = "dlc"
OUTPUT_METDATA_FILE = "output_metadata.json"

# output and preview filenames
DLC_MODELS_OUTPUT_FILE = "model.zip"
EVALUATION_RESULTS_OUTPUT_FILE = "evaluation_results.zip"
EVALUATION_RESULTS_PREVIEW_FILE = "evaluation_results_preview.svg"
EVALUATION_RESULTS_ERROR_DIST_PREVIEW_FILE = (
    "evaluation_results_error_distribution_preview.svg"
)
EVALUATION_RESULTS_ERROR_DIST_PCUTOFF_PREVIEW_FILE = (
    "evaluation_results_error_distribution_pcutoff_preview.svg"
)
EVALUATION_RESULTS_PLOTS_TRAIN_MOVIE_PREVIEW_FILE = (
    "evaluation_results_plots_train_movie_preview.mp4"
)
EVALUATION_RESULTS_PLOTS_TEST_MOVIE_PREVIEW_FILE = (
    "evaluation_results_plots_test_movie_preview.mp4"
)
EVALUATION_RESULTS_MAPS_LOCREF_MOVIE_PREVIEW_FILE = (
    "evaluation_results_maps_locref_movie_preview.mp4"
)
EVALUATION_RESULTS_MAPS_SCMAP_MOVIE_PREVIEW_FILE = (
    "evaluation_results_maps_scmap_movie_preview.mp4"
)
LEARNING_STATS_PREVIEW_FILE = "learning_stats_preview.svg"

# types of errors calculated for evaluation results
error_types_dict = {
    "tensorflow": {
        "Train Error": " Train error(px)",
        "Test Error": " Test error(px)",
        "Test Error w/ p-cutoff": "Train error with p-cutoff",
        "Train Error w/ p-cutoff": "Train error with p-cutoff",
    },
    "pytorch": {
        "Train Error": "train rmse",
        "Test Error": "test rmse",
        "Test Error w/ p-cutoff": "train rmse_pcutoff",
        "Train Error w/ p-cutoff": "test rmse_pcutoff",
    },
}

logger = logging.getLogger()


def _get_model_name(config: dict, trainingsetindex: int, shuffle: int):
    """Get the model name folder based on a dlc config fille.

    :param config: DLC config file read as a dictionary
    :param trainingsetindex: The index of the training fraction key in the DLC config file to use for the training fraction.
    :param shuffle: The shuffle number to use.
    """

    return f"{config['Task']}{config['date']}-trainset{int(config['TrainingFraction'][trainingsetindex] * 100.0)}shuffle{shuffle}"


def _get_dlc_scorer(
    config: dict, shuffle: int, engine: str, snapshot: Optional[int] = None
):
    """Get the name of the dlc scorer based on a dlc config fille.

    :param config: DLC config file read as a dictionary
    :param shuffle: The shuffle number to use.
    :param engine: The backend engine to use for training, either pytorch or tensorflow.
    :param snapshot: The snapshot number to use.
    """

    if engine == "pytorch":
        return f"DLC_{config['default_net_type'].replace('_', '').capitalize()}_{config['Task']}{config['date']}shuffle{shuffle}{'' if snapshot is None else f'_snapshot_{str(snapshot).zfill(3)}'}"
    else:
        return f"DLC_{config['default_net_type'].replace('_', '')}_{config['Task']}{config['date']}shuffle{shuffle}_{snapshot}"


def _get_evaluation_results_dir_name(engine: str):
    """Get the name of the evaluation results folder based on the engine.

    :param engine: The backend engine to use for training, either pytorch or tensorflow.
    """
    return f"evaluation-results{'-pytorch' if engine == 'pytorch' else ''}"


def _get_dlc_models_dir_name(engine: str):
    """Get the name of the dlc models folder based on the engine.

    :param engine: The backend engine to use for training, either pytorch or tensorflow.
    """
    return f"dlc-models{'-pytorch' if engine == 'pytorch' else ''}"


def _calculate_optimal_snapshot(
    evaluation_results_df: pd.DataFrame,
    optimal_snapshot_metric: str,
    engine: str,
):
    """Calculate the optimal snapshot from evaluation results.

    :param evaluation_results_df: The evaluation results as a dataframe.
    :param optimal_snapshot_metric: The error type to minimize to find the optimal snapshot.
    :param engine: The backend engine to use for training, either pytorch or tensorflow.
    """

    errors = evaluation_results_df[
        error_types_dict[engine][optimal_snapshot_metric]
    ].values
    x = "Training epochs" if engine == "pytorch" else "Training iterations:"

    rank = rankdata(errors, method="ordinal")
    optimal_ind = np.where(rank == 1)[0][0]
    return evaluation_results_df[x][optimal_ind], optimal_ind


def _generate_evaluation_results_preview(
    evaluation_results_df: pd.DataFrame,
    optimal_snapshot: int,
    optimal_snapshot_metric: str,
    engine: str,
    preview_filename: str,
):
    """Generate a preview for evaluation results

    :param evaluation_results_df: The evaluation results as a dataframe.
    :param optimal_snapshot: The optimal snapshot metric.
    :param optimal_snapshot_metric: The error type to minimize to find the optimal snapshot.
    :param engine: The backend engine to use for training, either pytorch or tensorflow.
    :param preview_filename: The name of the preview file.
    """

    df_x = "Training epochs" if engine == "pytorch" else "Training iterations:"
    data_x = (
        "Training Epochs" if engine == "pytorch" else "Training Iterations"
    )
    data = {data_x: [], "Error Type": [], "Error (px)": []}
    for i, iteration in enumerate(evaluation_results_df[df_x]):
        data[data_x] += [iteration] * (4)
        for name, error_type in error_types_dict[engine].items():
            data["Error Type"] += [name]
            data["Error (px)"] += [
                evaluation_results_df.iloc[i, :][error_type]
            ]

    df = pd.DataFrame(data)
    df_wide = df.pivot(index=data_x, columns="Error Type", values="Error (px)")

    ax = sns.lineplot(
        df_wide,
        palette=sns.color_palette("winter", len(error_types_dict[engine])),
    )
    optimal_df = evaluation_results_df.query(f"`{df_x}` == {optimal_snapshot}")
    ax.scatter(
        [[optimal_df[df_x]]],
        [[optimal_df[error_types_dict[engine][optimal_snapshot_metric]]]],
        label="Optimal Snapshot",
        color="deeppink",
    )
    ax.legend()

    plt.suptitle("Evaluation Results: Error vs. Iterations")
    plt.ylabel("RMSE (px)")
    plt.xticks(evaluation_results_df[df_x])
    plt.savefig(preview_filename, transparent=True)

    plt.clf()


def _pairwisedistances(
    DataCombined: pd.DataFrame,
    scorer1: str,
    scorer2: str,
    pcutoff: float = -1,
    bodyparts: Optional[List[str]] = None,
):
    """Calculates the pairwise Euclidean distance metric over body parts vs. images

    Copied from: https://github.com/DeepLabCut/DeepLabCut/blob/dd0ef5a2189eb1a2e8c29762784a5df259a6c3b8/deeplabcut/pose_estimation_tensorflow/core/evaluate.py#L24
    """
    mask = DataCombined[scorer2].xs("likelihood", level=1, axis=1) >= pcutoff
    if bodyparts is None:
        Pointwisesquareddistance = (
            DataCombined[scorer1] - DataCombined[scorer2]
        ) ** 2
        RMSE = np.sqrt(
            Pointwisesquareddistance.xs("x", level=1, axis=1)
            + Pointwisesquareddistance.xs("y", level=1, axis=1)
        )  # Euclidean distance (proportional to RMSE)
        return RMSE, RMSE[mask]
    else:
        Pointwisesquareddistance = (
            DataCombined[scorer1][bodyparts] - DataCombined[scorer2][bodyparts]
        ) ** 2
        RMSE = np.sqrt(
            Pointwisesquareddistance.xs("x", level=1, axis=1)
            + Pointwisesquareddistance.xs("y", level=1, axis=1)
        )  # Euclidean distance (proportional to RMSE)
        return RMSE, RMSE[mask]


def _calculate_evaluation_results_error_distribution(
    config: dict,
    optimal_snapshot: int,
    trainingsetindex: int,
    shuffle: int,
    engine: str,
    comparisonbodyparts: Optional[List[str]],
):
    """Calculate a error distribution of the optimal snapshot from evaluation results.

    :param config: DLC config file read as a dictionary
    :param optimal_snapshot: The optimal snapshot metric.
    :param trainingsetindex: The index of the training fraction key in the DLC config file to use for the training fraction.
    :param shuffle: The shuffle number to use.
    :param engine: The backend engine to use for training, either pytorch or tensorflow.
    :param comparisonbodyparts: The name of the preview file.
    """
    project_path = config["project_path"]
    iteration = config["iteration"]
    model_name = _get_model_name(
        config=config, trainingsetindex=trainingsetindex, shuffle=shuffle
    )
    dlc_scorer = _get_dlc_scorer(
        config=config,
        shuffle=shuffle,
        engine=engine,
        snapshot=optimal_snapshot,
    )
    # read collected data
    # get this from training-datasets
    collected_data_path = os.path.join(
        project_path,
        "training-datasets",
        f"iteration-{iteration}",
        f"UnaugmentedDataSet_{config['Task']}{config['date']}",
        f"CollectedData_{config['scorer']}.h5",
    )
    logger.info(f"Reading collected data from: {collected_data_path}")
    Data = pd.read_hdf(collected_data_path)

    # read model data
    # in evaluation results get file for optimal snapshot and load data
    model_results_path = os.path.join(
        project_path,
        _get_evaluation_results_dir_name(engine),
        f"iteration-{iteration}",
        model_name,
        (
            f"{dlc_scorer}.h5"
            if engine == "pytorch"
            else f"{dlc_scorer}-snapshot-{optimal_snapshot}.h5"
        ),
    )
    logger.info(f"Reading model results data from: {model_results_path}")
    DataMachine = pd.read_hdf(model_results_path)
    if engine == "pytorch":
        DataMachine.columns = DataMachine.columns.droplevel(
            level="individuals"
        )

    # _guarantee_multiindex_rows(Data)
    DataCombined = pd.concat([Data.T, DataMachine.T], axis=0, sort=False).T

    comparisonbodyparts = dlc.auxiliaryfunctions.intersection_of_body_parts_and_ones_given_by_user(
        config, comparisonbodyparts
    )
    logger.info(
        f"Calculating pairwise distances for the following body parts: {comparisonbodyparts}"
    )
    RMSE, RMSEpcutoff = _pairwisedistances(
        DataCombined,
        config["scorer"],
        dlc_scorer,
        config["pcutoff"],
        comparisonbodyparts,
    )

    RMSE.to_hdf(
        os.path.join(
            project_path,
            _get_evaluation_results_dir_name(engine),
            f"iteration-{iteration}",
            f"{dlc_scorer}_error_distribution.h5",
        ),
        "df_with_missing",
    )
    RMSEpcutoff.to_hdf(
        os.path.join(
            project_path,
            _get_evaluation_results_dir_name(engine),
            f"iteration-{iteration}",
            f"{dlc_scorer}_error_distribution_pcutoff.h5",
        ),
        "df_with_missing",
    )
    return RMSE, RMSEpcutoff


def _generate_evaluation_results_error_distribution_preview(
    config: dict,
    RMSE: np.array,
    RMSEpcutoff: np.array,
    comparisonbodyparts: str,
    preview_filename: str,
    pcutoff_preview_filename: str,
):
    """Generate previews for the error distribution of the optimal snapshot from evaluation results.

    :param config: DLC config file read as a dictionary
    :paran RMSE: The root mean squared error for the images evaluated with the optimal snapshot.
    :paran RMSE: The root mean squared error, filtered by p-cutoff, for the images evaluated with the optimal snapshot.
    :param comparisonbodyparts: The body parts to evaluate.
    :param preview_filename: The name of the preview for the RMSE.
    :param pcutoff_preview_filename: The name of the preview for the RMSE with p-cutoff.
    """
    comparisonbodyparts = dlc.auxiliaryfunctions.intersection_of_body_parts_and_ones_given_by_user(
        config, comparisonbodyparts
    )
    n = len(RMSE)
    data = {"bodyparts": [], "RMSE": [], "RMSE (w/ p-cutoff)": []}
    for i, bodypart in enumerate(comparisonbodyparts):
        data["RMSE"] += RMSE.iloc[:, i].to_list()
        data["RMSE (w/ p-cutoff)"] += RMSEpcutoff.iloc[:, i].to_list()
        data["bodyparts"] += [bodypart] * n
    df = pd.DataFrame(data)

    sns.set_style("white")

    plt.figure(figsize=(15, 10))
    sns.displot(
        df,
        x="RMSE",
        hue="bodyparts",
        kde=True,
        palette=sns.color_palette("spring", len(comparisonbodyparts)),
    )
    plt.suptitle("Optimal Snapshot Error Distribution")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(preview_filename, transparent=True)

    plt.clf()

    plt.figure(figsize=(15, 10))
    sns.displot(
        df,
        x="RMSE (w/ p-cutoff)",
        hue="bodyparts",
        kde=True,
        palette=sns.color_palette("summer", len(comparisonbodyparts)),
    )
    plt.suptitle("Optimal Snapshot Error Distribution (w/ p-cutoff)")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(pcutoff_preview_filename, transparent=True)

    plt.clf()


def _generate_images_movie_preview(
    image_files: List[str], preview_filename: str, sampling_rate: float = 1.0
):
    """Generate a movie preview from a series of images.

    :param image_files: List of image filenames to use.
    :param preview_filename: The filename of the movie preview.
    :param sampling_rate: The sampling rate of the movie.
    """

    max_resolution = (0, 0)
    for f in image_files:
        image = cv2.imread(f)

        if (
            image.shape[0] > max_resolution[0]
            and image.shape[1] > max_resolution[1]
        ):
            max_resolution = (image.shape[0], image.shape[1])

    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(
        preview_filename,
        fourcc,
        sampling_rate,
        (max_resolution[1], max_resolution[0]),
    )

    for f in image_files:
        image = cv2.imread(f)
        if (
            image.shape[0] != max_resolution[0]
            or image.shape[1] != max_resolution[1]
        ):
            image = cv2.resize(image, max_resolution)

        writer.write(image)

    writer.release()


def _generate_evaluation_results_plots_movie_preview(
    config: dict,
    optimal_snapshot: int,
    trainingsetindex: int,
    shuffle: int,
    engine: str,
    train_preview_filename: str,
    test_preview_filename: str,
):
    """Generate a movie preview for the plots generated from the evaluation results.

    :param config: DLC config file read as a dictionary
    :param optimal_snapshot: The optimal snapshot metric.
    :param trainingsetindex: The index of the training fraction key in the DLC config file to use for the training fraction.
    :param shuffle: The shuffle number to use.
    :param comparisonbodyparts: The name of the preview file.
    :param train_preview_filename: Filename of movie preview for train images.
    :param test_preview_filename: Filename of movie preview for test images.
    """

    project_path = config["project_path"]
    iteration = config["iteration"]
    model_name = _get_model_name(
        config=config, trainingsetindex=trainingsetindex, shuffle=shuffle
    )
    dlc_scorer = _get_dlc_scorer(
        config=config,
        shuffle=shuffle,
        engine=engine,
        snapshot=optimal_snapshot,
    )

    plots_dir = os.path.join(
        project_path,
        _get_evaluation_results_dir_name(engine),
        f"iteration-{iteration}",
        model_name,
        (
            f"LabeledImages_{dlc_scorer}"
            if engine == "pytorch"
            else f"LabeledImages_{dlc_scorer}_snapshot-{optimal_snapshot}"
        ),
    )

    image_files = os.listdir(plots_dir)
    train_image_files = [
        os.path.join(plots_dir, f)
        for f in image_files
        if f.startswith("Training-")
    ]
    test_image_files = [
        os.path.join(plots_dir, f)
        for f in image_files
        if f.startswith("Test-")
    ]

    _generate_images_movie_preview(train_image_files, train_preview_filename)

    _generate_images_movie_preview(test_image_files, test_preview_filename)


def _generate_evaluation_results_maps_movie_preview(
    config: dict,
    trainingsetindex: int,
    shuffle: int,
    engine: str,
    locref_preview_filename: str,
    scmap_preview_filename: str,
):
    """Generate a movie preview for the maps generated from the evaluation results.

    :param config: DLC config file read as a dictionary
    :param trainingsetindex: The index of the training fraction key in the DLC config file to use for the training fraction.
    :param shuffle: The shuffle number to use.
    :param engine: The backend engine to use for training, either pytorch or tensorflow.
    :param locref_preview_filename: Filename of movie preview for locref maps.
    :param scmap_preview_filename: Filename of movie preview for scoremaps.
    """

    project_path = config["project_path"]
    iteration = config["iteration"]
    model_name = _get_model_name(
        config=config, trainingsetindex=trainingsetindex, shuffle=shuffle
    )

    maps_dir = os.path.join(
        project_path,
        _get_evaluation_results_dir_name(engine),
        f"iteration-{iteration}",
        model_name,
        "maps",
    )

    image_files = os.listdir(maps_dir)
    locref_image_files = [
        os.path.join(maps_dir, f) for f in image_files if "locref" in f
    ]
    scmap_image_files = [
        os.path.join(maps_dir, f) for f in image_files if "scmap" in f
    ]

    _generate_images_movie_preview(locref_image_files, locref_preview_filename)

    _generate_images_movie_preview(scmap_image_files, scmap_preview_filename)


def _generate_learning_stats_preview(
    config: dict,
    trainingsetindex: int,
    shuffle: int,
    engine: str,
    preview_filename: str,
):
    """Generate a preview for the learning stats generated during model training.

    :param config: DLC config file read as a dictionary
    :param trainingsetindex: The index of the training fraction key in the DLC config file to use for the training fraction.
    :param shuffle: The shuffle number to use.
    :param engine: The backend engine to use for training, either pytorch or tensorflow.
    :param preview_filename: Filename of the preview.
    """
    project_path = config["project_path"]
    iteration = config["iteration"]
    model_name = _get_model_name(
        config=config, trainingsetindex=trainingsetindex, shuffle=shuffle
    )
    learning_stats_file = os.path.join(
        project_path,
        _get_dlc_models_dir_name(engine),
        f"iteration-{iteration}",
        model_name,
        "train",
        "learning_stats.csv",
    )
    logger.info(f"Reading learning stats from: {learning_stats_file}")
    if engine == "pytorch":
        df = pd.read_csv(
            learning_stats_file,
        )

        df = df[["step", "losses/train.total_loss"]]
        df = df.rename(
            columns={"step": "Epoch", "losses/train.total_loss": "Loss"}
        )

        # load pytorch to get learning rate schedule
        pytorch_config_file = os.path.join(
            project_path,
            _get_dlc_models_dir_name(engine),
            f"iteration-{iteration}",
            model_name,
            "train",
            "pytorch_config.yaml",
        )

        with open(pytorch_config_file, "r") as f:
            pytorch_config = yaml.safe_load(f)

        schedule_epochs = pytorch_config["runner"]["scheduler"]["params"][
            "milestones"
        ]
        schedule_learning_rates = pytorch_config["runner"]["scheduler"][
            "params"
        ]["lr_list"]
        schedule_learning_rates = [x[0] for x in schedule_learning_rates]

        initial_learning_rate = pytorch_config["runner"]["optimizer"][
            "params"
        ]["lr"]
        schedule_epochs = [1] + schedule_epochs
        schedule_learning_rates = [
            initial_learning_rate
        ] + schedule_learning_rates

        learning_rates = []
        for epoch in df["Epoch"]:
            for i in range(len(schedule_epochs)):
                if i == len(schedule_epochs) - 1:
                    learning_rates.append(schedule_learning_rates[i])
                    break

                if (
                    epoch >= schedule_epochs[i]
                    and epoch < schedule_epochs[i + 1]
                ):
                    learning_rates.append(schedule_learning_rates[i])
                    break
        df["Learning Rate"] = learning_rates

        ax = sns.scatterplot(
            df,
            x="Epoch",
            y="Loss",
            hue="Learning Rate",
            palette=sns.color_palette(
                "cool", len(df["Learning Rate"].unique())
            ),
        )

        ax.plot(
            df["Epoch"],
            df["Loss"],
            label="Loss",
            color="gray",
            alpha=0.5,
        )
    else:
        df = pd.read_csv(
            learning_stats_file,
            header=None,
            names=["Training Iteration", "Loss", "Learning Rate"],
        )
        ax = sns.scatterplot(
            df,
            x="Training Iteration",
            y="Loss",
            hue="Learning Rate",
            palette=sns.color_palette(
                "cool", len(df["Learning Rate"].unique())
            ),
        )

        ax.plot(
            df["Training Iteration"],
            df["Loss"],
            label="Loss",
            color="gray",
            alpha=0.5,
        )

    ax.set_title("Learning Stats")
    plt.tight_layout()

    plt.savefig(preview_filename, transparent=True)
    plt.clf()


def _save_output_metadata(
    config: dict,
    engine: str,
    output_dlc_files: List[str],
    output_movie_files: List[str],
    optimal_snapshot: int,
    evaluation_results_df: dict,
):
    """Save output metadata for tool.

    :param config: DLC config file read as a dictionary.
    :param engine: The backend engine to use for training, either pytorch or tensorflow.
    :param output_dlc_files: List of output files to attach dlc metadata to.
    :param output_movie_files: List of output movies to generate metadata for.
    :param optimal_snapshot: The optimal snapshot number:
    :param evaluation_results_df: Evaluation results of the training.
    """

    logger.info("Getting config file")
    keys = []
    output_metadata = {}
    for file in output_movie_files:
        basename, _ = os.path.splitext(os.path.basename(file))
        output_metadata[basename] = (
            get_timing_spacing_metadata_from_labeled_movie(file)
        )

    for file in output_dlc_files:
        basename, _ = os.path.splitext(os.path.basename(file))
        output_metadata[basename] = {}
        keys.append(basename)

    median_error_dict = {
        "snapshot_median_train_error": f"{np.median(evaluation_results_df[error_types_dict[engine]['Train Error']]):.3f}",
        "snapshot_median_train_pcutoff_error": f"{np.median(evaluation_results_df[error_types_dict[engine]['Train Error w/ p-cutoff']]):.3f}",
        "snapshot_median_test_error": f"{np.median(evaluation_results_df[error_types_dict[engine]['Test Error']]):.3f}",
        "snapshot_median_test_pcutoff_error": f"{np.median(evaluation_results_df[error_types_dict[engine]['Test Error w/ p-cutoff']]):.3f}",
    }

    for key in keys:
        output_metadata[key][DLC_METADATA_KEY] = {}

        output_metadata[key][DLC_METADATA_KEY]["task"] = config["Task"]
        output_metadata[key][DLC_METADATA_KEY]["scorer"] = config["scorer"]
        output_metadata[key][DLC_METADATA_KEY]["date"] = config["date"]
        output_metadata[key][DLC_METADATA_KEY]["multi_animal_project"] = (
            config["multianimalproject"]
        )
        output_metadata[key][DLC_METADATA_KEY]["model_neural_net_type"] = (
            config["default_net_type"]
        )
        output_metadata[key][DLC_METADATA_KEY]["body_parts"] = ",".join(
            config[
                (
                    "multianimalbodyparts"
                    if config["multianimalproject"]
                    else "bodyparts"
                )
            ]
        )
        output_metadata[key][DLC_METADATA_KEY]["snapshot"] = str(
            optimal_snapshot
        )

        for k, v in median_error_dict.items():
            output_metadata[key][DLC_METADATA_KEY][k] = v

    with open(OUTPUT_METDATA_FILE, "w") as f:
        json.dump(output_metadata, f, indent=4)


def update_pose_cfg_file(
    pose_cfg_file: str,
    output_file: str,
    type: str,
    config: dict,
    engine: str,
    trainingsetindex: int,
    shuffle: int,
):
    """Update pose config file based on user input"""

    project_path = config["project_path"]
    iteration = config["iteration"]
    model_name = _get_model_name(
        config=config, trainingsetindex=trainingsetindex, shuffle=shuffle
    )
    actual_pose_cfg_file = os.path.join(
        project_path,
        _get_dlc_models_dir_name(engine),
        f"iteration-{iteration}",
        model_name,
        type,
        output_file,
    )

    logger.info(
        f"Copying input {type} pose cfg file {pose_cfg_file} to model {type} pose cfg file {actual_pose_cfg_file}"
    )

    with open(pose_cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    with open(actual_pose_cfg_file, "r") as f:
        actual_cfg = yaml.safe_load(f)

    if "project_path" in actual_cfg:
        cfg["project_path"] = actual_cfg["project_path"]

    if "dataset" in actual_cfg:
        cfg["dataset"] = actual_cfg["dataset"]

    if "metadataset" in actual_cfg:
        cfg["metadataset"] = actual_cfg["metadataset"]

    if "metadata" in actual_cfg:
        if "project_path" in actual_cfg["metadata"]:
            cfg["metadata"]["project_path"] = actual_cfg["metadata"][
                "project_path"
            ]

        if "pose_config_path" in actual_cfg["metadata"]:
            cfg["metadata"]["pose_config_path"] = actual_cfg["metadata"][
                "pose_config_path"
            ]

    with open(actual_pose_cfg_file, "w") as f:
        yaml.dump(cfg, f)

    logger.info(f"Updated {type} pose cfg file")


def train_model(
    labeled_data_zip_file: List[str],
    config_file: List[str],
    engine: Optional[str] = None,
    training_fraction: Optional[float] = None,
    p_cutoff: Optional[float] = None,
    train_files: Union[List[str], str] = "auto",
    test_files: Union[List[str], str] = "auto",
    net_type: str = "resnet_50",
    augmenter_type: str = "default",
    train_pose_cfg_file: Optional[List[str]] = None,
    pytorch_cfg_file: Optional[List[str]] = None,
    num_snapshots_to_evaluate: str = "all",
    save_iters: int = 50000,
    max_iters: int = 1030000,
    allow_growth: bool = True,
    keepdeconvweights: bool = True,
    device: str = "auto",
    batch_size: int = 4,
    epochs: int = 200,
    save_epochs: int = 50,
    detector_batch_size: int = None,
    detector_epochs: int = None,
    detector_save_epochs: int = None,
    pose_threshold: float = 0.1,
    test_pose_cfg_file: Optional[List[str]] = None,
    plot_evaluation_results: bool = False,
    body_parts_to_evaluate: str = "all",
    rescale: bool = False,
    per_keypoint_evaluation: bool = False,
    generate_maps: bool = False,
    optimal_snapshot_metric: str = "Test Error",
    save_optiomal_snapshot: bool = True,
    output_dir: str = "",
):
    """Train a DLC model.

    :param labeled_data_zip_file: A zip file containing the labeled data to use for training.
    :param config_file: A config file describing the configuration of the DLC project.
        Includes the body parts and skeleton of the animal pose,
        Training fraction, network and augmenter types, etc.
    :param engine: Specifies the backend engine to use for training the network.
        Either "tensorflow" or "pytorch".
        If empty, reads the engine value in the input config.yaml.
        DeepLabCut recommends using pytorch since it is faster.
        The tensorflow backend will be deprecated by the end of 2024.
    :param training_fraction: The fraction of labeled data to use for training the network,
        the remainder is used for testing the network for validation.
        This value is automatically calculated if train_file and test_files are specified.
        If empty, and no train_files and test_files are specified,
        then the value in the input config file is used.
    :param p_cutoff: Cutoff threshold for the confidence of model results.
        If used, then results with a confidence below the cutoff are omitted from analysis.
        This is used when generating evaluation results, and can be used downstream to filter
        analyses on novel videos.
    :param train_files: Comma-separated list of video file names from labeled data to use for training the network.
        Must be specified if test_files is passed.
        If empty, then train images are automatically selected from labeled data.
    :param test_files: Comma-separated list of video file names from labeled data to use for training the network.
        Must be specified if train_files is passed.
        If empty, then test images are automatically selected from labeled data.
    :param net_type: Type of network. Currently supported options are: resnet_50, resnet_101, resnet_152,
        mobilenet_v2_1.0, mobilenet_v2_0.75, mobilenet_v2_0.5, mobilenet_v2_0.35,
        efficientnet-b0, efficientnet-b, efficientnet-b2, efficientnet-b3, efficientnet-b4,
        efficientnet-b5, efficientnet-b6
    :param augemnter_type: Type of augmenter.
        Currently supported options are: default, scalecrop, imgaug, tensorpack, deterministic
    :param train_pose_cfg_file: A config file describing the configuration settings for training the network.
    :param num_snapshots_to_evaluate: Sets how many snapshots are evaluated, i.e. states of the trained network.
        Every save iteration period, a snapshot is stored, however only the last num_snapshots_to_evaluate are kept.
        If empty, then all snapshots are evaluated.
    :param save_iters: The interval to save an iteration as a snapshot.
        If empty, then the value in pose_config.yaml is used.
    :param max_iters: The maximum number of iterations to train the model.
        If empty, then the value in pose_config.yaml is used.
    :param allow_growth: For some smaller GPUs, memory issues happen.
        If true then, the memory allocator does not pre-allocate the entire specified GPU memory region,
        instead starting small and growing as needed.
        See issue: https://forum.image.sc/t/how-to-stop-running-out-of-vram/30551/2
    :param keepdeconvweights: Restores the weights of the deconvolution layers (and the backbone) when training from a snapshot.
        Note that if you change the number of bodyparts, you need to set this to false for re-training.
    :param device: (pytorch only) The torch device to train on (such as "cpu", "cuda", "mps", "auto"). By default set to "auto".
    :param batch_size: (pytorch only) Overrides the batch size to train with in pytorch_config.yaml
        Small batch sizes (i.e., value of one) is good for training on a CPU, but this tool runs on a GPU instance.
        For GPUs a larger batch size should be used.
        The value should be the biggest power of 2 where you don't geta CUDA out-of-memory error, such as 8, 16, 32 or 64.
        By default a batch size of 4 is used as this has shown to stay within memory limits on the instance used for this tool.
        This also allows you to increase the learning rate (empirically you can scale the learning rate by sqrt(batch_size) times).
    :param epochs. (pytorch only) Overrides the maximum number of epochs to train the model for.
    :param save_epochs: (pytorch only) Overrides the number of epochs between each snapshot save.
    :param detector_batch_size: (pytorch only) Only for top-down models. Overrides the batch size with
        which to train the detector.
    :param detector_epochs: (pytorch only) Only for top-down models. Overrides the maximum number of
        epochs to train the model for. Setting to 0 means the detector will not be trained.
    :param detector_save_epochs: (pytorch only) Only for top-down models. Overrides the number of epochs
            between each snapshot of the detector is saved.
    :param pose_threshold: used for memory-replay. pseudo predictions that are below this are discarded for memory-replay
    :param test_pose_cfg_file: A config file describing the configuration settings for testing (evaluating) the network.
    :param plot_evaluation_results: Plots the predictions of evaluation results on the train and test images.
    :param body_parts_to_evaluate: The average error for evaluation results will be computed for those body parts only.
        The provided list has to be a subset of the defined body parts.
        Otherwise, set to “all” to compute error over all body parts.
    :param rescale: Evaluate the model at the 'global_scale' variable (as set in the pose_config.yaml file for a particular project).
        I.e. every image will be resized according to that scale and prediction will be compared to the resized ground truth.
        The error will be reported in pixels at rescaled to the original size.
        I.e. For a [200,200] pixel image evaluated at global_scale=.5, the predictions are calculated on [100,100] pixel images,
        compared to 1/2*ground truth and this error is then multiplied by 2.
        The evaluation images are also shown for the original size.
    :param per_keypoint_evaluation: Compute the train and test RMSE for each keypoint,
        and save the results to a {model_name}-keypoint-results.csv in the evalution-results folder
    :param generate_maps: Plot the scoremaps, locref layers, and PAFs from the evaluation results.
    :param optimal_snapshot_metric: The type of error from the evaluation results to minimize in order to find the optimal snapshot.
        Options are: Train Error, Test Error, Train Error w/ p-cutoff, Test Error w/ p-cutoff
    :param save_optiomal_snapshot: Only keep the snapshot with the smallest error from the evaluation results in the zipped model output.
        The error to minimize is indicated by the optimal_snapshot_metric parameter.
        This significantly reduces the size of the zipped model output uploaded to IDEAS.
        If set to false, then all snapshots evaluated are included in the zipped model output.
        This is controlled by the num_snapshots_to_evaluate parameter.
    :param output_dir: Output directory (set during testing only.)
        If empty, use cwd.
    """
    logger.info("Starting DLC model training")

    # only get a single file for each input
    labeled_data_zip_file = labeled_data_zip_file[0]
    config_file = config_file[0]

    if not output_dir:
        output_dir = os.getcwd()

    # hardcorded parameters
    # we only train a single network / shuffle
    num_shuffles = 1
    shuffle = 1
    trainingsetindex = 0

    # create an empty dlc project
    project_dir = output_dir
    logger.info("Creating new project to place model in")
    tmp_config_file = dlc.create_new_project(
        project="Project",
        experimenter="IDEAS",
        # dlc requires list of videos when creating a new project,
        # but they're not used at all after labeled data is generated.
        # just provide any mp4 movies in data dir.
        videos=[
            "/ideas/data/2023-01-27-10-34-22-camera-1_trimmed_1s_dlc_labeled_movie.mp4"
        ],
        working_directory=project_dir,
        copy_videos=False,
    )

    # copy input config to new project
    # but first get project path
    config = dlc.auxiliaryfunctions.read_config(tmp_config_file)
    project_path = config["project_path"]
    logger.info(f"Project created at: {project_path}")

    config = dlc.auxiliaryfunctions.read_config(config_file)
    config["project_path"] = project_path

    # copy other input config values if provided
    logger.info("Updating input config with tool parameters for config")
    if engine:
        logger.info(f"Updating engine in config file: {engine}")
        config["engine"] = engine
    else:
        assert (
            "engine" in config
        ), "No engine specified in config file or input parameters"
        engine = config["engine"]

    if training_fraction:
        logger.info(
            f"Updating training fraction in config file: {training_fraction}"
        )
        config["TrainingFraction"] = [training_fraction]

    if p_cutoff:
        logger.info(f"Updating p-cutoff in config file: {p_cutoff}")
        config["pcutoff"] = p_cutoff

    if num_snapshots_to_evaluate == "all":
        num_snapshots_to_evaluate = None
    else:
        try:
            num_snapshots_to_evaluate = int(num_snapshots_to_evaluate)
        except ValueError:
            raise IdeasError(
                f"Failed to convert num_snapshots_to_evaluate({num_snapshots_to_evaluate}) to integer"
            )

    # update config file
    logger.info("Copying input config file to project")
    config_file = tmp_config_file
    dlc.auxiliaryfunctions.write_config(config_file, config)

    # extract labeled data into new project
    logger.info("Extracting input labeled data to project")
    labeled_data_dir = os.path.join(project_path, "labeled-data")
    shutil.rmtree(labeled_data_dir)
    with zipfile.ZipFile(labeled_data_zip_file, "r") as z:
        logger.info(f"Contents of labeled data zip: {z.namelist()}")
        if "labeled-data/" in z.namelist():
            labeled_data_dir = project_path
        z.extractall(labeled_data_dir)

    # determine train and test indices if provided
    train_indices = None
    test_indices = None
    if train_files != "auto" and test_files != "auto":
        if isinstance(train_files, str):
            train_files = [train_files]

        if isinstance(test_files, str):
            test_files = [test_files]

        assert isinstance(
            train_files, list
        ), "Train files must be a list of video names"
        assert isinstance(
            test_files, list
        ), "Test files must be a list of video names"
        logger.info(
            "User provided custom train/test split. Generating train/test indices"
        )

        if set(train_files).intersection(set(test_files)):
            raise IdeasError(
                "List of train files cannot intersect with test files"
            )

        # Dummy training dataset to get indexes and so on from later
        dlc.create_training_dataset(
            config=config_file,
            num_shuffles=num_shuffles,
        )

        # get list of all images
        collected_data_file = os.path.join(
            project_path,
            "training-datasets",
            f"iteration-{config['iteration']}",
            f"UnaugmentedDataSet_{config['Task']}{config['date']}",
            f"CollectedData_{config['scorer']}.h5",
        )
        collected_data_df = pd.read_hdf(collected_data_file)

        # match images to train/test split
        image_paths = (
            collected_data_df.index.to_list()
        )  # turn dataframe into list
        if train_files:
            train_indices = []
            for i, path in enumerate(image_paths):
                if str(path[1]) in train_files:
                    train_indices.append(i)

            logger.info(f"Using train indices: {train_indices}")

        if test_files:
            test_indices = []
            for i, path in enumerate(image_paths):
                if str(path[1]) in test_files:
                    test_indices.append(i)

            logger.info(f"Using test indices: {test_indices}")

        # delete previous training set and model folders for dummy data
        logger.info("Removing dummy training set")
        shutil.rmtree(
            os.path.join(
                project_path,
                "training-datasets",
            )
        )
        shutil.rmtree(
            os.path.join(
                project_path,
                _get_dlc_models_dir_name(engine),
            )
        )

        # calculate training fraction and update config file
        train_fraction = round(
            len(train_indices)
            * 1.0
            / (len(train_indices) + len(test_indices)),
            2,
        )
        config["TrainingFraction"] = [train_fraction]
        logger.info(
            f"Updating training fraction in config.yaml for custom train/test split: {train_fraction}"
        )
        dlc.auxiliaryfunctions.write_config(config_file, config)

        train_indices = [train_indices]
        test_indices = [test_indices]

    # create training dataset
    # https://github.com/DeepLabCut/DeepLabCut/blob/v2.3.10/deeplabcut/generate_training_dataset/trainingsetmanipulation.py#L769
    logger.info("Creating training datasets")
    dlc.create_training_dataset(
        config=config_file,
        num_shuffles=num_shuffles,
        Shuffles=None,
        windows2linux=False,
        userfeedback=False,
        trainIndices=train_indices,
        testIndices=test_indices,
        net_type=net_type,
        augmenter_type=augmenter_type,
        posecfg_template=None,
        superanimal_name="",
        weight_init=None,
    )

    if train_pose_cfg_file is not None:
        train_pose_cfg_file = train_pose_cfg_file[0]
        logger.info("Updating train pose cfg file from input file")
        update_pose_cfg_file(
            train_pose_cfg_file,
            "pose_cfg.yaml",
            "train",
            config,
            engine,
            trainingsetindex,
            shuffle,
        )

    if test_pose_cfg_file is not None:
        test_pose_cfg_file = test_pose_cfg_file[0]
        logger.info("Updating test pose cfg file from input file")
        update_pose_cfg_file(
            test_pose_cfg_file,
            "pose_cfg.yaml",
            "test",
            config,
            engine,
            trainingsetindex,
            shuffle,
        )

    if pytorch_cfg_file is not None:
        pytorch_cfg_file = pytorch_cfg_file[0]
        logger.info("Updating pytorch cfg file from input file")
        update_pose_cfg_file(
            pytorch_cfg_file,
            "pytorch_config.yaml",
            "train",
            config,
            engine,
            trainingsetindex,
            shuffle,
        )

    # train the network
    # https://github.com/DeepLabCut/DeepLabCut/blob/v2.3.10/deeplabcut/pose_estimation_tensorflow/training.py
    logger.info("Training the network")
    dlc.train_network(
        config=config_file,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        max_snapshots_to_keep=num_snapshots_to_evaluate,
        displayiters=None,
        saveiters=save_iters,
        maxiters=max_iters,
        allow_growth=allow_growth,
        gputouse=GPU_TO_USE,
        autotune=False,
        keepdeconvweights=keepdeconvweights,
        modelprefix="",
        superanimal_name="",
        superanimal_transfer_learning=False,
        engine=None,
        device=device,
        snapshot_path=None,
        detector_path=None,
        batch_size=batch_size,
        epochs=epochs,
        save_epochs=save_epochs,
        detector_batch_size=detector_batch_size,
        detector_epochs=detector_epochs,
        detector_save_epochs=detector_save_epochs,
        pose_threshold=pose_threshold,
    )

    # evaluate the network
    # https://github.com/DeepLabCut/DeepLabCut/blob/v2.3.10/deeplabcut/pose_estimation_tensorflow/core/evaluate.py
    logger.info("Evaluating the network")
    logger.info("Updating snapshotindex to all for evaluation:")
    config["snapshotindex"] = "all"
    dlc.auxiliaryfunctions.write_config(config_file, config)
    dlc.evaluate_network(
        config=config_file,
        Shuffles=[shuffle],
        trainingsetindex=trainingsetindex,
        plotting=plot_evaluation_results,
        show_errors=True,
        comparisonbodyparts=body_parts_to_evaluate,
        gputouse=GPU_TO_USE,
        rescale=rescale,
        modelprefix="",
        per_keypoint_evaluation=per_keypoint_evaluation,
        engine=None,
        device=device,
        transform=None,
        detector_snapshot_index=None,
    )

    logger.info("Reading evaluation results")
    # read evaluation results in order to calculate optimal snapshot
    evaluation_results_df = pd.read_csv(
        os.path.join(
            project_path,
            _get_evaluation_results_dir_name(engine),
            "iteration-0",
            "CombinedEvaluation-results.csv",
        )
    )
    logger.info(f"Got evaluation results: {evaluation_results_df}")

    optimal_snapshot, optimal_snapshot_ind = _calculate_optimal_snapshot(
        evaluation_results_df,
        optimal_snapshot_metric=optimal_snapshot_metric,
        engine=engine,
    )
    logger.info(f"Calculated optimal snapshot: {optimal_snapshot}")

    # generate maps from evaluation results
    if generate_maps:
        if engine == "pytorch":
            logger.warning("Cannot generate maps for pytorch engine")
        else:
            logger.info("Generating maps for optimal snapshot")
            logger.info(
                "Extracting plots of the scoremaps, locref layers, and PAFs from evaluation results"
            )
            logger.info(
                f"Updating snapshotindex to optimal snapshot {optimal_snapshot_ind} for maps"
            )
            config["snapshotindex"] = int(optimal_snapshot_ind)
            dlc.auxiliaryfunctions.write_config(config_file, config)
            dlc.extract_save_all_maps(
                config=config_file, shuffle=shuffle, Indices=None
            )

    # generate error distribution of optimal snapshot
    logger.info("Generating error distribution for optimal snapshot")
    try:
        RMSE, RMSEpcutoff = _calculate_evaluation_results_error_distribution(
            config,
            optimal_snapshot,
            trainingsetindex=trainingsetindex,
            shuffle=shuffle,
            engine=engine,
            comparisonbodyparts=body_parts_to_evaluate,
        )
    except Exception as error:
        logger.warning(
            f"Failed to generate evaluation results error distribution: {str(error)}"
        )
        logger.exception(error)

    # update snapshot index for output model
    logger.info(
        "Updating snapshot index in model output to point to default last snapshot in train dir"
    )
    config["snapshotindex"] = -1
    dlc.auxiliaryfunctions.write_config(config_file, config)

    # copy some evaluation results to output model
    logger.info("Copying summary evaluation results to model zip output")
    try:
        iteration = config["iteration"]

        dlc_scorer = _get_dlc_scorer(
            config=config,
            shuffle=shuffle,
            engine=engine,
            snapshot=optimal_snapshot,
        )
        files = [
            os.path.join(
                project_path,
                _get_evaluation_results_dir_name(engine),
                f"iteration-{iteration}",
                f"{dlc_scorer}_error_distribution.h5",
            ),
            os.path.join(
                project_path,
                _get_evaluation_results_dir_name(engine),
                f"iteration-{iteration}",
                f"{dlc_scorer}_error_distribution_pcutoff.h5",
            ),
            os.path.join(
                project_path,
                _get_evaluation_results_dir_name(engine),
                f"iteration-{iteration}",
                "CombinedEvaluation-results.csv",
            ),
        ]
        for f in files:
            shutil.copy(f, project_path)
    except Exception as error:
        logger.warning(
            f"Failed to save summary evaluation to model zip output"
        )
        logger.exception(error)

    # save output model to zip file
    logger.info("Saving trained model to model zip output")
    model_filename = os.path.join(output_dir, DLC_MODELS_OUTPUT_FILE)
    with zipfile.ZipFile(model_filename, "w", zipfile.ZIP_DEFLATED) as f:
        f.write(os.path.join(project_path, "config.yaml"), "config.yaml")
        f.write(
            os.path.join(project_path, "CombinedEvaluation-results.csv"),
            "CombinedEvaluation-results.csv",
        )
        f.write(
            os.path.join(
                project_path,
                f"{dlc_scorer}_error_distribution.h5",
            ),
            f"{dlc_scorer}_error_distribution.h5",
        )
        f.write(
            os.path.join(
                project_path,
                f"{dlc_scorer}_error_distribution_pcutoff.h5",
            ),
            f"{dlc_scorer}_error_distribution_pcutoff.h5",
        )

        snapshot_name = (
            f"snapshot-{str(optimal_snapshot).zfill(3)}"
            if engine == "pytorch"
            else f"snapshot-{optimal_snapshot}"
        )
        for root, dirs, files in os.walk(
            os.path.join(project_path, _get_dlc_models_dir_name(engine))
        ):
            for name in files:
                if (
                    save_optiomal_snapshot
                    and "snapshot" in name
                    and snapshot_name not in name
                ):
                    continue

                file_path = os.path.join(root, name)
                rel_path = file_path[len(project_path) + 1 :]
                f.write(file_path, rel_path)

    # save evaluation results to zip file
    evaluation_results_filename = os.path.join(
        output_dir, os.path.splitext(EVALUATION_RESULTS_OUTPUT_FILE)[0]
    )
    shutil.make_archive(
        evaluation_results_filename,
        format="zip",
        root_dir=project_path,
        base_dir=_get_evaluation_results_dir_name(engine),
    )

    # track movie previews generated for output metadata
    movie_files = []

    # generate previews
    logger.info("Generating evaluation results preview")
    try:
        preview_filename = os.path.join(
            output_dir, EVALUATION_RESULTS_PREVIEW_FILE
        )
        _generate_evaluation_results_preview(
            evaluation_results_df,
            optimal_snapshot,
            optimal_snapshot_metric,
            engine,
            preview_filename,
        )
    except Exception as error:
        logger.warning(
            f"Failed to generate evaluation results preview: {error}"
        )
        logger.exception(error)

    logger.info(
        "Generating previews for evaluation results error distributions"
    )
    try:
        preview_filename = os.path.join(
            output_dir, EVALUATION_RESULTS_ERROR_DIST_PREVIEW_FILE
        )
        pcutoff_preview_filename = os.path.join(
            output_dir, EVALUATION_RESULTS_ERROR_DIST_PCUTOFF_PREVIEW_FILE
        )
        _generate_evaluation_results_error_distribution_preview(
            config,
            RMSE,
            RMSEpcutoff,
            body_parts_to_evaluate,
            preview_filename,
            pcutoff_preview_filename,
        )
    except Exception as error:
        logger.warning(
            f"Failed to generate evaluation results error distribution preview"
        )
        logger.exception(error)

    if plot_evaluation_results:
        logger.info(
            "Generating previews for evaluation results plots movie preview"
        )
        try:
            train_preview_filename = os.path.join(
                output_dir, EVALUATION_RESULTS_PLOTS_TRAIN_MOVIE_PREVIEW_FILE
            )
            test_preview_filename = os.path.join(
                output_dir, EVALUATION_RESULTS_PLOTS_TEST_MOVIE_PREVIEW_FILE
            )
            _generate_evaluation_results_plots_movie_preview(
                config,
                optimal_snapshot,
                trainingsetindex,
                shuffle,
                engine,
                train_preview_filename,
                test_preview_filename,
            )
            movie_files.append(train_preview_filename)
            movie_files.append(test_preview_filename)
        except Exception as error:
            logger.warning(
                f"Failed to generate evaluation results plots movie preview"
            )
            logger.exception(error)

    if generate_maps and engine != "pytorch":
        logger.info(
            "Generating previews for evaluation results maps movie preview"
        )
        try:
            locref_preview_filename = os.path.join(
                output_dir, EVALUATION_RESULTS_MAPS_LOCREF_MOVIE_PREVIEW_FILE
            )
            scmap_preview_filename = os.path.join(
                output_dir, EVALUATION_RESULTS_MAPS_SCMAP_MOVIE_PREVIEW_FILE
            )
            _generate_evaluation_results_maps_movie_preview(
                config,
                trainingsetindex,
                shuffle,
                engine,
                locref_preview_filename,
                scmap_preview_filename,
            )
            movie_files.append(locref_preview_filename)
            movie_files.append(scmap_preview_filename)
        except Exception as error:
            logger.warning(
                f"Failed to generate evaluation results maps movie preview"
            )
            logger.exception(error)

    logger.info("Generating learning stats preview")
    try:
        preview_filename = os.path.join(
            output_dir, LEARNING_STATS_PREVIEW_FILE
        )
        _generate_learning_stats_preview(
            config, trainingsetindex, shuffle, engine, preview_filename
        )
    except Exception as error:
        logger.warning(f"Failed to generate learning stats preview")
        logger.exception(error)

    logger.info("Generating output metadata")
    try:
        _save_output_metadata(
            config,
            engine,
            [model_filename, evaluation_results_filename],
            movie_files,
            evaluation_results_df=evaluation_results_df,
            optimal_snapshot=optimal_snapshot,
        )
    except Exception as error:
        logger.warning(f"Failed to generate output metadata")
        logger.exception(error)

    logger.info("Finished DLC model training")
