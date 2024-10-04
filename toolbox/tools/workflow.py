import json
import logging
import os
import shutil
import time
import zipfile
from glob import glob
from pathlib import Path
from typing import List, Optional
import yaml
import deeplabcut as dlc
from deeplabcut.core.engine import Engine
import isx
import matplotlib
import numpy as np
from ideas.exceptions import IdeasError
from toolbox.utils.io import convert_dlc_h5_to_annotations
from toolbox.utils.utilities import (
    count_video_frames,
    get_timing_spacing_metadata_from_labeled_movie,
    order_movie_files,
    update_config_file,
)

logger = logging.getLogger()

# ----------- CONSTANTS ------------
# default name of model config file
MODEL_CONFIG_FILE = "config.yaml"
DLC_MODELS_DIR = "dlc-models"
POSE_CFG_FILE = "pose_cfg.yaml"
PYTORCH_CFG_FILE = "pytorch_config.yaml"

EXPERIMENT_ANNOTATIONS_FORMATS = ("parquet", "csv")
SUPPORTED_FILE_FORMATS = (
    "mp4",
    "avi",
    "isxb",
)

DLC_OUTPUT_FILE = "dlc_pose_estimates"
ANNOTATIONS_OUTPUT_FILE = "dlc_annotations"
MOVIE_OUTPUT_FILE = "dlc_labeled_movie"
PREVIEW_HIST_FILE = "hist"
PREVIEW_PLOT_LIKELIHOOD = "plot-likelihood"
PREVIEW_PLOT = "plot"
PREVIEW_TRAJECTORY = "trajectory"
PREVIEWS = [
    PREVIEW_HIST_FILE,
    PREVIEW_PLOT_LIKELIHOOD,
    PREVIEW_PLOT,
    PREVIEW_TRAJECTORY,
]
PREVIEW_EXT = "png"
PREVIEW_DIR = "plot-poses"

DLC_METADATA_KEY = "dlc"
OUTPUT_METDATA_FILE = "output_metadata.json"


def _get_model_name(config: dict, trainingsetindex: int, shuffle: int):
    """Get the model name folder based on a dlc config fille.

    :param config: DLC config file read as a dictionary
    :param trainingsetindex: The index of the training fraction key in the DLC config file to use for the training fraction.
    :param shuffle: The shuffle number to use.
    """

    return f"{config['Task']}{config['date']}-trainset{int(config['TrainingFraction'][trainingsetindex] * 100.0)}shuffle{shuffle}"


def _update_train_pose_cfg(
    dict_train: dict, dict2change: dict, saveasfile: str
):
    """Updates pose_cfg.yaml in the train dir.

    Used for creating a project from a user-uploaded model.
    This function was copied from: https://github.com/DeepLabCut/DeepLabCut/blob/main/deeplabcut/create_project/modelzoo.py#L44

    :param dict dict_train: The current contents of pose_cfg.yaml.
    :param dict dict2change: The keys to update in pose_cfg.yaml.
    :param str saveasfile: The path to pose_cfg.yaml where the changes are saved.
    """
    for key in dict2change.keys():
        dict_train[key] = dict2change[key]
    dlc.auxiliaryfunctions.write_plainconfig(saveasfile, dict_train)


def _create_test_pose_cfg(dictionary, keys2save, saveasfile):
    """Creates pose_cfg.yaml in the test dir.

    Used for creating a project from a user-uploaded model.
    This function was copied from: https://github.com/DeepLabCut/DeepLabCut/blob/main/deeplabcut/create_project/modelzoo.py#L50

    :param dict dictionary: The contents of pose_cfg.yaml in the train dir.
    :param list keys2save: The keys to copy from the train pose_cfg.yaml to the test pose_cfg.yaml.
    :param str saveasfile: The path to pose_cfg.yaml where the changes are saved.
    """
    dict_test = {}
    for key in keys2save:
        dict_test[key] = dictionary[key]
    dict_test["scoremap_dir"] = "test"
    dict_test["global_scale"] = 1.0
    dlc.auxiliaryfunctions.write_plainconfig(saveasfile, dict_test)


def _get_snapshot_nums(snapshot_dir, recursive=False):
    """Gets a list of unique snapshot numbers from a dir where snapshots are stored.

    :param str snapshot_dir: The directory where snapshots are stored.
    :param bool recursive: Whether to search recursively in the snapshot dir.
    """
    snapshot_nums = set()
    search_str = (
        f"{snapshot_dir}/**/snapshot*"
        if recursive
        else f"{snapshot_dir}/snapshot*"
    )
    for filename in glob(search_str, recursive=recursive):
        _, basename = os.path.split(filename)
        start_index = basename.find("-")
        end_index = basename.find(".")
        snapshot_num = int(basename[start_index + 1 : end_index])
        snapshot_nums.add(snapshot_num)
    return list(snapshot_nums)


def _validate_one_snapshot(snapshot_dir):
    """Validates that only one snapshot is stored in a user-uploaded model.

    The size of snapshot files can be quite large, on the order of 100s of MBs.
    This function will check the number of unique snapshot numbers in the snapshot dir.
    If more than one unique snapshot number is detected, then a warning is logged
    so that the user is aware they should decrease the size of their uploaded model dir to save on storage.

    :param str snapshot_dir: The directory where snapshots are stored.
    """
    snapshot_nums = _get_snapshot_nums(snapshot_dir, recursive=False)
    if len(snapshot_nums) > 1:
        logger.warn(
            f"Found {len(snapshot_nums)} snapshots. Consider removing unnecessary snapshots from the model dir before uploading to IDEAS in order to save on storage."
        )
    elif len(snapshot_nums) == 0:
        logger.warn("No snapshots found!")
    else:
        logger.info("Only one snapshot in the dir. Looks good.")


def _validate_or_transform_model_dir(
    model_dir, movie_files, output_dir=None, use_existing=False
):
    """Validates a user-uploaded zipped model, or transforms it to a model dir that can be used by DLC.

    DLC has specific requirements for the structure & contents of a model dir, otherwise execution will fail.
    This function will detect if the contents of the zip file fulfill the requirements of a valid
    DLC model dir. If not, then the function will attempt to create an empty model dir and move the relevant
    model files to this dir so that it can be used by DLC.

    :param str model_dir: The model dir to validate or tranform.
    :param list movie_files: The movie files to pass to DLC if creating a new model dir.
    :param str output_dir: The output directory to store the model
    :param bool use_existing: Flag indicating whether to use an existing project if it exists,
        or create a new project and copy snapshot files from the input model.
    """
    logger.info("Validating contents of unzipped model dir")

    # check if model_dir contains config.yaml and dlc-models folder
    actual_config_file = os.path.join(model_dir, MODEL_CONFIG_FILE)
    exists = os.path.exists(actual_config_file)
    if exists and use_existing:
        # if yes, verify contents of sub dirs
        logger.info(
            "Top-level of model dir is in expected format for DeepLabCut. Validatng contents of sub-dirs"
        )
        config = dlc.auxiliaryfunctions.read_config(actual_config_file)

        # grab relevant values from config.yaml that determine sub-dir names
        iteration = int(config["iteration"])
        task = config["Task"]
        date = config["date"]
        training_fraction = int(config["TrainingFraction"][0] * 100.0)

        dlc_models_dir = os.path.join(model_dir, DLC_MODELS_DIR)
        iteration_dir = os.path.join(dlc_models_dir, f"iteration-{iteration}")
        logger.info(os.listdir(dlc_models_dir))
        if not os.path.exists(iteration_dir):
            raise IdeasError(
                f"Could not find iteration-{iteration} dir within {DLC_MODELS_DIR} dir"
            )

        logger.info(
            f"Found iteration-{iteration} dir within {DLC_MODELS_DIR} dir"
        )
        model_name = f"{task}{date}-trainset{training_fraction}shuffle1"
        trainset_dir = os.path.join(iteration_dir, model_name)
        if not os.path.exists(trainset_dir):
            raise IdeasError(
                f"Could not find {model_name} dir within iteration-{iteration} dir"
            )

        logger.info(f"Found {model_name} dir within iteration-{iteration} dir")
        test_dir = os.path.join(trainset_dir, "test")
        train_dir = os.path.join(trainset_dir, "train")
        if not os.path.exists(test_dir) or not os.path.exists(train_dir):
            raise IdeasError(
                f"Could not find test or train dir within {model_name} dir"
            )

        logger.info(f"Found test and train dirs within {model_name} dir")
        if not os.path.exists(
            os.path.join(test_dir, POSE_CFG_FILE)
        ) or not os.path.exists(os.path.join(train_dir, POSE_CFG_FILE)):
            raise IdeasError(
                f"Could not find {POSE_CFG_FILE} within test or train dirs"
            )

        logger.info(f"Found {POSE_CFG_FILE} within test and train dirs")
        _validate_one_snapshot(train_dir)
        return model_dir, False
    else:
        # if not, then search for dir which contains snapshot* files
        logger.info(
            "Model dir is not expected format for DeepLabCut, searching for snapshot files recursively in this dir."
        )
        snapshot_dir = None
        for filename in glob(f"{model_dir}/**/snapshot*", recursive=True):
            file_dir, _ = os.path.split(filename)
            logger.info(f"Found snapshot file: {filename}, in dir: {file_dir}")
            if snapshot_dir:
                if snapshot_dir != file_dir:
                    raise IdeasError(
                        "Snapshots found in multiple directories, cannot determine which one to use. Place snapshot to use in one directory and remove others."
                    )
            else:
                snapshot_dir = file_dir
        _validate_one_snapshot(snapshot_dir)

        if snapshot_dir is None:
            raise IdeasError("Failed to find any snapshots in zipped model.")

        # detect
        engine = (
            "pytorch"
            if "pytorch_config.yaml" in os.listdir(snapshot_dir)
            else "tensorflow"
        )
        logger.info(f"Contents of snapshot_dir: {os.listdir(snapshot_dir)}")
        logger.info(f"Detected engine {engine}")

        logger.info(
            f"Found snapshot dir: {snapshot_dir}. Verifying {POSE_CFG_FILE} exists in this dir too."
        )
        # assert that there is a pose_cfg.yaml file in the same dir
        # if not, then raise an error
        # otherwise create a new project and organize the dir
        pose_cfg_file = os.path.join(snapshot_dir, POSE_CFG_FILE)
        if not os.path.exists(pose_cfg_file):
            raise IdeasError(
                f"No {POSE_CFG_FILE} found in snapshot dirctory '{snapshot_dir}'"
            )

        if engine == "pytorch":
            logger.info(
                f"Verifying {PYTORCH_CFG_FILE} exists in this dir too."
            )
            pytorch_cfg_file = os.path.join(snapshot_dir, PYTORCH_CFG_FILE)
            if not os.path.exists(pytorch_cfg_file):
                raise IdeasError(
                    f"No {PYTORCH_CFG_FILE} found in snapshot dirctory '{snapshot_dir}'"
                )

        pose_cfg = dlc.auxiliaryfunctions.read_plainconfig(pose_cfg_file)
        pose_cfg["dataset_type"] = "imgaug"

        # create new project folder
        logger.info("Creating new project to place model in")
        config_file = dlc.create_new_project(
            project="Project",
            experimenter="IDEAS",
            videos=movie_files,
            working_directory=model_dir if output_dir is None else output_dir,
            copy_videos=False,
        )

        # Create test and train dirs
        config = dlc.auxiliaryfunctions.read_config(config_file)

        if exists:
            logger.info("Copying input config to new project")
            shutil.copy(actual_config_file, config_file)
            update_config_file(
                config_file=config_file,
                updates={"project_path": config["project_path"]},
            )
        else:
            # update config file based on pose cfg
            logger.info("Updating config file based on pose cfg")
            update_config_file(
                config_file=config_file,
                updates={
                    "default_net_type": pose_cfg["net_type"],
                    "default_augmenter": pose_cfg["dataset_type"],
                    "bodyparts": pose_cfg["all_joints_names"],
                    "dotsize": 6,
                    "engine": engine,
                },
            )

        # Create test and train dirs
        config = dlc.auxiliaryfunctions.read_config(config_file)
        train_dir = Path(
            os.path.join(
                config["project_path"],
                str(
                    dlc.auxiliaryfunctions.get_model_folder(
                        trainFraction=config["TrainingFraction"][0],
                        shuffle=1,
                        cfg=config,
                        engine=(
                            Engine.PYTORCH
                            if engine == "pytorch"
                            else Engine.TF
                        ),
                    )
                ),
                "train",
            )
        )
        test_dir = Path(
            os.path.join(
                config["project_path"],
                str(
                    dlc.auxiliaryfunctions.get_model_folder(
                        trainFraction=config["TrainingFraction"][0],
                        shuffle=1,
                        cfg=config,
                        engine=(
                            Engine.PYTORCH
                            if engine == "pytorch"
                            else Engine.TF
                        ),
                    )
                ),
                "test",
            )
        )

        logger.info("Creating train and test model directories")
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        modelfoldername = dlc.auxiliaryfunctions.get_model_folder(
            trainFraction=config["TrainingFraction"][0],
            shuffle=1,
            cfg=config,
            engine=Engine.PYTORCH if engine == "pytorch" else Engine.TF,
        )
        path_train_config = str(
            os.path.join(
                config["project_path"],
                Path(modelfoldername),
                "train",
                "pose_cfg.yaml",
            )
        )
        path_test_config = str(
            os.path.join(
                config["project_path"],
                Path(modelfoldername),
                "test",
                "pose_cfg.yaml",
            )
        )
        path_pytorch_config = str(
            os.path.join(
                config["project_path"],
                Path(modelfoldername),
                "train",
                "pytorch_config.yaml",
            )
        )

        # move snapshot files to new model dir
        for filename in glob(f"{snapshot_dir}/snapshot*"):
            _, basename = os.path.split(filename)
            new_filename = os.path.join(train_dir, basename)
            logger.info(
                f"Copying snapshot file '{filename}' to '{new_filename}'"
            )
            shutil.copy(filename, new_filename)

        # move pose_cfg.yaml to new model dir
        shutil.copy(pose_cfg_file, path_train_config)

        if engine == "pytorch":
            shutil.copy(pytorch_cfg_file, path_pytorch_config)

        # make some final updates to pose_cfg.yaml in test and train dirs
        dict2change = {
            "project_path": str(config["project_path"]),
        }

        _update_train_pose_cfg(pose_cfg, dict2change, path_train_config)
        keys2save = [
            "dataset",
            "dataset_type",
            "num_joints",
            "all_joints",
            "all_joints_names",
            "net_type",
            "init_weights",
            "global_scale",
            "location_refinement",
            "locref_stdev",
        ]
        _create_test_pose_cfg(pose_cfg, keys2save, path_test_config)

        return config["project_path"], True, engine


def _extract_pretrained_model_dir(
    model_dir: str, movie_files: List[str], output_dir: str
):
    """Extract a pre-trained model dir from a .zip file.

    :param str model_dir: The path to the .zip file.
    :param list movie_files: The input movies which may be passed to DLC
        when initializing an empty model dir.
    :param output_dir str: The output directory where the contents of the .zip file are extracted.
    """
    logger.info("Model dir is a zip file. Extracting now...")

    model_dir_name = os.path.splitext(os.path.basename(model_dir))[0]
    tmp_model_dir = f"{output_dir}/{model_dir_name}"

    logger.info(
        f"Extracting contents of zipped model dir to '{tmp_model_dir}'"
    )
    with zipfile.ZipFile(model_dir, "r") as zip_ref:
        zip_ref.extractall(tmp_model_dir)

    model_dir = tmp_model_dir
    logger.info("Contents of unzipped model dir:")
    logger.info(os.listdir(f"{model_dir}"))

    model_dir, is_new_project, engine = _validate_or_transform_model_dir(
        model_dir=model_dir, movie_files=movie_files
    )
    logger.info(f"Transformed model dir: {model_dir}")
    logger.info(os.listdir(f"{model_dir}"))

    return model_dir, is_new_project, engine


def _create_training_dataset_metadata(
    config_file: str, trainingsetindex: int, shuffle: int, engine: str
):
    """Create training dataset metadata.

    This metadata file is required to exists and be populated with
    critical metadata describing the configuration of the model to train.

    :param config_file: The config file of the project.
    :param trainingsetindex: The training set index of the model.
    :param shuffle: The shuffle index of the model.
    :param engine: The backend engine to use for training, either pytorch or tensorflow.
    """

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    project_path = config["project_path"]
    training_set_dir = os.path.join(
        project_path,
        "training-datasets",
        f"iteration-{config['iteration']}",
        f"UnaugmentedDataSet_{config['Task']}{config['date']}",
    )
    metadata_file = os.path.join(training_set_dir, "metadata.yaml")

    model_name = _get_model_name(
        config=config, trainingsetindex=trainingsetindex, shuffle=shuffle
    )
    train_fraction = config["TrainingFraction"][trainingsetindex]
    metadata = {
        "shuffles": {
            model_name: {
                "train_fraction": train_fraction,
                "index": shuffle,
                "split": 1,
                "engine": engine,
            }
        }
    }
    with open(metadata_file, "w") as f:
        yaml.dump(metadata, f)
        logger.info(f"metadata: {metadata}")

    with open(metadata_file, "r") as f:
        metadata = yaml.safe_load(f)
        logger.info(f"metadata: {metadata}")


def _write_output_metadata_workflow(
    model_dir: str,
    output_movie_files: List[str],
    output_pose_estimate_files: List[str],
    is_new_project: bool,
):
    """Helper function for writing output metadata for the workflow tool.

    Writes output metadata for the three outputs of the workflow tool:
    * DLC output .h5 file
    * Experiment annotations file
    * Labeled movie .mp4 file

    :param model_dir: The model directory. Output metadata about the metadata
        will be extracted from the config file.
    :param output_movie_files: The output movie file to extract timing and spacing
        metadata from.
    """
    logger.info("Getting config file")
    config_file = os.path.join(model_dir, MODEL_CONFIG_FILE)
    if not os.path.exists(config_file):
        raise IdeasError(
            f"Model config file ({config_file}) does not exist",
        )

    config = dlc.utils.auxiliaryfunctions.read_config(config_file)
    keys = [ANNOTATIONS_OUTPUT_FILE]
    output_metadata = {ANNOTATIONS_OUTPUT_FILE: {}}
    for file in output_movie_files:
        basename, _ = os.path.splitext(os.path.basename(file))
        output_metadata[
            basename
        ] = get_timing_spacing_metadata_from_labeled_movie(file)
        keys.append(basename)

    for file in output_pose_estimate_files:
        basename, _ = os.path.splitext(os.path.basename(file))
        output_metadata[basename] = {}
        keys.append(basename)

    for key in keys:
        output_metadata[key][DLC_METADATA_KEY] = {}

        output_metadata[key][DLC_METADATA_KEY]["task"] = config["Task"]
        output_metadata[key][DLC_METADATA_KEY]["scorer"] = config["scorer"]
        output_metadata[key][DLC_METADATA_KEY]["date"] = config["date"]

        output_metadata[key][DLC_METADATA_KEY][
            "multi_animal_project"
        ] = config["multianimalproject"]
        output_metadata[key][DLC_METADATA_KEY][
            "model_neural_net_type"
        ] = config["default_net_type"]
        output_metadata[key][DLC_METADATA_KEY]["body_parts"] = ",".join(
            config[
                (
                    "multianimalbodyparts"
                    if config["multianimalproject"]
                    else "bodyparts"
                )
            ]
        )
        output_metadata[key][DLC_METADATA_KEY][
            "snapshot"
        ] = _get_snapshot_nums(model_dir, recursive=True)[
            config["snapshotindex"]
        ]

    with open(OUTPUT_METDATA_FILE, "w") as f:
        json.dump(output_metadata, f, indent=4)


def run_workflow(
    *,
    model_dir: List[str],
    movie_files: List[str],
    experiment_annotations_format: str = "parquet",
    crop_rect: dict = [],
    window_length: int = 5,
    displayed_body_parts: str = "all",
    p_cutoff: float = 0.6,
    dot_size: int = 5,
    color_map: str = "rainbow",
    keypoints_only: bool = False,
    output_frame_rate: Optional[int] = None,
    draw_skeleton: bool = False,
    trail_points: int = 0,
    output_dir: str = "",
):
    """Run a DeepLabCut workflow on behaviorial movies.

    The workflow consists of three steps:
    1. Analyze videos using the input model
    2. Filter results
    3. Label videos with results

    :param str model_dir: Path to the DLC model directory.
        If the path is to a .zip file, the model directory will be extracted
        and used for execution with DeepLabCut.
    :param str output_dir: Path to the output directory.
    :param List[str] movie_files: Behavioural movies to analyze.
        Must be one of the following formats: .isxb, .mp4, and .avi.
    :param str experiment_annotations_format: The file format of the output experiment annotations file.
        Can be either .parquet or .csv
    :param dict crop_rect: ROI input representing the crop rect area.
        The crop rect is represented as:
        [{
            "groupKey": "crop_rect",
            "top": ...,
            "left": ...,
            "width": ...,
            "height": ....
        }]
        If empty, then no cropping is applied.
    :param int window_length: Length of the median filter to apply on the model results.
        Must be an odd number. If zero, then no filtering is applied.
    :param str displayed_body_parts: Selects the body parts that are plotted in the video.
        If all, then all body parts from config.yaml are used.
    :param float p_cutoff: Cutoff threshold for predictions when labelling the
        input movie. If predictions are below the threshold then they are not displayed.
    :param int dot_size: Size in pixels to draw a point labelling a body part.
    :param str color_map: Color map used to color body part labels. Any matplotlib
        colormap name is acceptable.
    :param bool keypoints_only: Only display keypoints, not video frames.
    :param Optional[int] output_frame_rate: Positive number, output frame rate for
        labeled video. If None, use the input movie frame rate.
    :param bool draw_skeleton: If True adds a line connecting the body parts making
        a skeleton on each frame. The body parts to be connected and the color of these
        connecting lines are specified by the color_map.
    :param int trail_points: Number of previous frames whose body parts are plotted
        in a frame (for displaying history).
    """
    logger.info("Running DLC workflow")

    if not output_dir:
        output_dir = os.getcwd()

    logger.info("Validating inputs")

    input_file_extensions = [Path(f).suffix[1:].lower() for f in movie_files]
    matching_file_ext = all(
        ext in SUPPORTED_FILE_FORMATS for ext in input_file_extensions
    )
    unique_file_ext = np.unique(input_file_extensions).size == 1
    if not matching_file_ext or not unique_file_ext:
        raise IdeasError("Input file formats do not match.")

    model_dir = model_dir[0]
    tmp_model_dir = None
    if not os.path.isdir(model_dir):
        # is_new_project = True
        model_dir_name = os.path.splitext(os.path.basename(model_dir))[0]
        tmp_model_dir = f"{output_dir}/{model_dir_name}"
        model_dir, is_new_project, engine = _extract_pretrained_model_dir(
            model_dir=model_dir, movie_files=movie_files, output_dir=output_dir
        )
    else:
        is_new_project = False
        model_dir, _, engine = _validate_or_transform_model_dir(
            model_dir=model_dir, movie_files=movie_files, output_dir=output_dir
        )

    if experiment_annotations_format not in EXPERIMENT_ANNOTATIONS_FORMATS:
        raise IdeasError(
            f"Experiment annotations format ({experiment_annotations_format}) must be either {EXPERIMENT_ANNOTATIONS_FORMATS}."
        )

    config_file = os.path.join(model_dir, MODEL_CONFIG_FILE)
    if not os.path.exists(config_file):
        raise IdeasError(
            f"Model config file ({config_file}) does not exist",
        )

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        project_path = config["project_path"]

    try:
        if displayed_body_parts != "all":
            _displayed_body_parts = displayed_body_parts.split(",")
            config = dlc.utils.auxiliaryfunctions.read_config(config_file)
            if any(
                [
                    part not in config["bodyparts"]
                    for part in _displayed_body_parts
                ]
            ):
                raise IdeasError(
                    f"Displayed body parts contains an invalid body part name. Body parts of model are: {config['bodyparts']}"
                )
    except Exception as error:
        raise IdeasError(
            f"Failed to parse displayed body parts ({displayed_body_parts}). Expected format is the value 'all' or a comma-seperated list of body parts tracked by the model."
        ) from error

    if color_map not in matplotlib.colormaps():
        raise IdeasError(
            (
                f"Color map ({color_map}) must be one of the following "
                f"matplotlib color maps: {matplotlib.colormaps()}"
            ),
        )

    # calculate the total num frames, to benchmark performance
    num_frames = sum(
        [count_video_frames(movie_file) for movie_file in movie_files]
    )

    # determine if output is filtered, used for labeling
    is_filtered = window_length > 1

    # Dummy training dataset to get indexes and so on from later
    dlc.create_training_dataset(
        config=config_file,
        num_shuffles=1,
    )

    logger.info("Create training set metadata.yaml")
    _create_training_dataset_metadata(
        config_file=config_file, trainingsetindex=0, shuffle=1, engine=engine
    )

    analyze(
        model_dir=model_dir,
        output_dir=output_dir,
        movie_files=movie_files,
        crop_rect=crop_rect,
        num_frames=num_frames,
        plot_trajectories=(not is_filtered),
    )

    filter(
        model_dir=model_dir,
        output_dir=output_dir,
        movie_files=movie_files,
        window_length=window_length,
        num_frames=num_frames,
        plot_trajectories=is_filtered,
    )

    label(
        model_dir=model_dir,
        output_dir=output_dir,
        movie_files=movie_files,
        is_filtered=is_filtered,
        displayed_body_parts=displayed_body_parts,
        p_cutoff=p_cutoff,
        dot_size=dot_size,
        color_map=color_map,
        keypoints_only=keypoints_only,
        output_frame_rate=output_frame_rate,
        draw_skeleton=draw_skeleton,
        trail_points=trail_points,
        num_frames=num_frames,
    )

    logger.info("Renaming h5 outputs")
    pose_estimates_h5_file_ending = "_filtered.h5" if is_filtered else ".h5"
    pose_estimates_output_files = []
    for file in [
        file
        for file in os.listdir(output_dir)
        if file.lower().endswith(pose_estimates_h5_file_ending)
    ]:
        for i, movie_file in enumerate(movie_files):
            basename = os.path.splitext(os.path.basename(movie_file))[0]
            if file.startswith(basename):
                output_file = os.path.join(
                    output_dir, f"{DLC_OUTPUT_FILE}.{i}.h5"
                )
                logger.info(f"Renaming {file} to {DLC_OUTPUT_FILE}.{i}.h5")
                os.rename(os.path.join(output_dir, file), output_file)
                pose_estimates_output_files.append(output_file)

                for preview in PREVIEWS:
                    preview_file = os.path.join(
                        output_dir,
                        PREVIEW_DIR,
                        basename,
                        f"{preview}{'_filtered' if is_filtered else ''}.{PREVIEW_EXT}",
                    )
                    if os.path.exists(preview_file):
                        logger.info(
                            f"Renaming {preview_file} to {preview}.{i}.{PREVIEW_EXT}"
                        )
                        os.rename(
                            preview_file,
                            os.path.join(
                                output_dir, f"{preview}.{i}.{PREVIEW_EXT}"
                            ),
                        )

    experiment_annotations_file_path = f"{output_dir}/{ANNOTATIONS_OUTPUT_FILE}.{experiment_annotations_format}"
    logger.info(
        "Converting DeepLabCupt h5 output to experiment annotations output."
    )
    try:
        convert_dlc_h5_to_annotations(
            pose_estimates_h5_files=pose_estimates_output_files,
            movie_files=movie_files,
            annotations_file=experiment_annotations_file_path,
        )
    except Exception as error:
        logger.warn(
            f"Failed to convert DeepLabCut h5 output to experiment annotations object with error: {error}"
        )

    logger.info("Renaming mp4 outputs")
    labeled_mp4_output_files = []
    for file in [
        file
        for file in os.listdir(output_dir)
        if file.lower().endswith("_labeled.mp4")
    ]:
        for i, movie_file in enumerate(movie_files):
            basename = os.path.splitext(os.path.basename(movie_file))[0]
            if file.startswith(basename):
                output_file = os.path.join(
                    output_dir, f"{MOVIE_OUTPUT_FILE}.{i}.mp4"
                )
                logger.info(f"Renaming {file} to {MOVIE_OUTPUT_FILE}.{i}.mp4")
                os.rename(os.path.join(output_dir, file), output_file)
                labeled_mp4_output_files.append(output_file)

    logger.info("Writing output metadata")
    try:
        _write_output_metadata_workflow(
            model_dir,
            labeled_mp4_output_files,
            pose_estimates_output_files,
            is_new_project,
        )
    except Exception as error:
        logger.warn(f"Failed to generate output metadata with error: {error}")

    if tmp_model_dir:
        logger.info("Removing temporary model dir")
        shutil.rmtree(tmp_model_dir)

    if os.path.exists(project_path):
        logger.info("Removing DLC project")
        shutil.rmtree(project_path)

    logging.info("Finished running DLC workflow")


def analyze(
    *,
    model_dir: str,
    output_dir: str,
    movie_files: List[str],
    crop_rect: str = "",
    num_frames: Optional[int] = None,
    plot_trajectories: bool = False,
):
    """Analyze behaviorial movies with a pre-trained DeepLabCut model.

    :param str model_dir: Path to the DLC model directory.
    :param str output_dir: Path to the output directory.
    :param List[str] movie_files: Behavioural movies to analyze.
        Must be one of the following formats: .isxb, .mp4, and .avi.
    :param str crop_rect: List of 4 values representing the coordinates of the top-left and bottom-right corners
        of the movie FOV to crop prior to running the model. Formatted as a comma-seperated list of integers in
        the following order: top_left_x, bottom_right_x, top_left_y, bottom_right_y.
        If blank, then no cropping is applied.
    :param int num_frames: The number of frames in the input movie, to calculate FPS.
    """
    logger.info("Executing DeepLabCut model on input movies")

    movie_file_format = os.path.splitext(movie_files[0])[1][1:]
    if movie_file_format not in SUPPORTED_FILE_FORMATS:
        raise IdeasError(
            (
                f"Input movie files ({movie_files}) must be one "
                f"of the following formats: {SUPPORTED_FILE_FORMATS}"
            ),
        )

    logger.info("Getting config file")
    config_file = os.path.join(model_dir, MODEL_CONFIG_FILE)
    if not os.path.exists(config_file):
        raise IdeasError(
            f"Model config file ({config_file}) does not exist",
        )

    logger.info("Parsing crop rect str")
    try:
        _crop_rect = None
        if crop_rect:
            assert (
                len(crop_rect) == 1 and crop_rect[0]["groupKey"] == "crop_rect"
            )
            x_left = crop_rect[0]["left"]
            x_right = x_left + crop_rect[0]["width"]
            y_top = crop_rect[0]["top"]
            y_bottom = y_top + crop_rect[0]["height"]
            _crop_rect = [
                int(round(point))
                for point in [x_left, x_right, y_top, y_bottom]
            ]
            logger.info(
                f"Parsed crop rect roi input for DeepLabCut: {_crop_rect}"
            )
    except Exception as error:
        raise IdeasError(
            f"Failed to parse crop rect ({crop_rect}), unexpected format for ROI input."
        ) from error

    logger.info("nvidia-smi output:")
    logger.info(os.system("nvidia-smi"))

    if movie_file_format == "isxb":
        tmp_mp4_movie_files = [
            os.path.join(
                output_dir,
                os.path.splitext(os.path.basename(movie_file))[0] + ".mp4",
            )
            for movie_file in movie_files
        ]

        logger.info(
            f"Exporting input .isxb movie file ({movie_files}) "
            f"to mp4 file ({tmp_mp4_movie_files}) temporarily"
        )
        for isxb_file, mp4_file in zip(movie_files, tmp_mp4_movie_files):
            isx.export_movie_to_mp4(isxb_file, mp4_file)
        movie_files = tmp_mp4_movie_files

    gpu_to_use = int(os.getenv("GPU_TO_USE", 0))
    use_gpu = int(os.getenv("USE_GPU", 1))
    logger.info(f"Use gpu: {use_gpu}, gpu id: {gpu_to_use}")

    try:
        start = time.time()
        dlc.analyze_videos(
            config_file,
            movie_files,
            destfolder=output_dir,
            cropping=_crop_rect,
            gputouse=gpu_to_use if use_gpu else None,
        )
        end = time.time()

        if not (
            [
                file
                for file in os.listdir(output_dir)
                if file.lower().endswith(".h5")
            ]
        ):
            raise IdeasError("No h5 output generated by DeepLabCut.")

        logger.info(
            (
                "Finished executing DeepLabCut model with "
                f"overall performance of {num_frames / (end - start)} FPS"
            )
        )
    except Exception as error:
        raise IdeasError("Failed to execute DeepLabCut model") from error

    if plot_trajectories:
        logger.info(
            "Plotting trajectories from non-filtered data using DeepLabCut"
        )
        try:
            dlc.utils.plotting.plot_trajectories(
                config=config_file,
                videos=movie_files,
                destfolder=output_dir,
                filtered=False,
                showfigures=True,
            )
        except Exception as error:
            logger.warn(
                f"Failed to plot trajectories from non-filtered data using DeepLabCut with error: {error}"
            )


def filter(
    *,
    model_dir: str,
    output_dir: str,
    movie_files: List[str],
    window_length: int = 5,
    num_frames: Optional[int] = None,
    plot_trajectories: bool = False,
):
    """Filter DeepLabCut model results.

    :param str model_dir: Path to the DLC model directory.
    :param str output_dir: Path to the output directory.
    :param List[str] movie_files: Behavioural movies to analyze.
        Must be one of the following formats: .isxb, .mp4, and .avi.
    :param int num_frames: The number of frames in the input movie, to calculate FPS.
    """
    logger.info("Filtering DeepLabCut pose estimates")

    movie_file_format = os.path.splitext(movie_files[0])[1][1:]
    if movie_file_format not in SUPPORTED_FILE_FORMATS:
        raise IdeasError(
            (
                f"Input movie files ({movie_files}) must be one "
                f"of the following formats: {SUPPORTED_FILE_FORMATS}"
            ),
        )

    if movie_file_format == "isxb":
        tmp_mp4_movie_files = [
            os.path.join(
                output_dir,
                os.path.splitext(os.path.basename(movie_file))[0] + ".mp4",
            )
            for movie_file in movie_files
        ]
        movie_files = tmp_mp4_movie_files

    logger.info("Getting config file")
    config_file = os.path.join(model_dir, MODEL_CONFIG_FILE)
    if not os.path.exists(config_file):
        raise IdeasError(
            f"Model config file ({config_file}) does not exist",
        )

    is_filtered = window_length > 1
    if is_filtered:
        try:
            start = time.time()
            dlc.filterpredictions(
                config_file,
                movie_files,
                destfolder=output_dir,
                windowlength=window_length,
                filtertype="median",
                save_as_csv=False,
            )
            end = time.time()

            if not (
                [
                    file
                    for file in os.listdir(output_dir)
                    if file.lower().endswith("_filtered.h5")
                ]
            ):
                raise IdeasError(
                    "No filtered h5 output generated by DeepLabCut."
                )

            logger.info(
                (
                    "Finished filtering DeepLabCut pose estimates with "
                    f"overall performance of {num_frames / (end - start)} FPS"
                )
            )
        except Exception as error:
            raise IdeasError(
                "Failed to filter DeepLabCut pose estimates."
            ) from error

        if plot_trajectories:
            logger.info(
                "Plotting trajectories from filtered data using DeepLabCut"
            )
            try:
                dlc.utils.plotting.plot_trajectories(
                    config=config_file,
                    videos=movie_files,
                    destfolder=output_dir,
                    filtered=True,
                    showfigures=True,
                )
            except Exception as error:
                logger.warn(
                    f"Failed to plot trajectories from filtered data using DeepLabCut with error: {error}"
                )
    else:
        logger.info("Filter window length is 1. Skipping filtering.")


def label(
    *,
    model_dir: str,
    output_dir: str,
    movie_files: List[str],
    is_filtered: bool,
    displayed_body_parts: str = "all",
    p_cutoff: float = 0.6,
    dot_size: int = 5,
    color_map: str = "rainbow",
    keypoints_only: bool = False,
    output_frame_rate: Optional[int] = None,
    draw_skeleton: bool = False,
    trail_points: int = 0,
    num_frames: Optional[int] = None,
):
    """Label movies with DeepLabCut results.

    :param str model_dir: Path to the DLC model directory.
    :param str output_dir: Path to the output directory.
    :param List[str] movie_files: Behavioural movies to analyze.
        Must be one of the following formats: .isxb, .mp4, and .avi.
    :param str displayed_body_parts: Selects the body parts that are plotted in the video.
        If all, then all body parts from config.yaml are used.
    :param float p_cutoff: Cutoff threshold for predictions when labelling the
        input movie. If predictions are below the threshold then they are not displayed.
    :param int dot_size: Size in pixels to draw a point labelling a body part.
    :param str color_map: Color map used to color body part labels. Any matplotlib
        colormap name is acceptable.
    :param bool keypoints_only: Only display keypoints, not video frames.
    :param Optional[int] output_frame_rate: Positive number, output frame rate for
        labeled video. If None, use the input movie frame rate.
    :param bool draw_skeleton: If True adds a line connecting the body parts making
        a skeleton on each frame. The body parts to be connected and the color of these
        connecting lines are specified by the color_map.
    :param int trail_points: Number of previous frames whose body parts are plotted
        in a frame (for displaying history).
    :param int num_frames: The number of frames in the input movie, to calculate FPS.
    """
    logger.info("Labeling DeepLabCut pose estimates")

    movie_file_format = os.path.splitext(movie_files[0])[1][1:]
    if movie_file_format not in SUPPORTED_FILE_FORMATS:
        raise IdeasError(
            (
                f"Input movie files ({movie_files}) must be one "
                f"of the following formats: {SUPPORTED_FILE_FORMATS}"
            ),
        )

    if movie_file_format == "isxb":
        tmp_mp4_movie_files = [
            os.path.join(
                output_dir,
                os.path.splitext(os.path.basename(movie_file))[0] + ".mp4",
            )
            for movie_file in movie_files
        ]
        movie_files = tmp_mp4_movie_files

    logger.info("Getting config file")
    config_file = os.path.join(model_dir, MODEL_CONFIG_FILE)
    if not os.path.exists(config_file):
        raise IdeasError(
            f"Model config file ({config_file}) does not exist",
        )

    logger.info("Updating config file with labelling params")
    try:
        update_config_file(
            config_file=config_file,
            updates={
                "pcutoff": p_cutoff,
                "dotsize": dot_size,
                "colormap": color_map,
            },
        )
    except Exception as error:
        raise IdeasError(
            "Failed to update config.yaml file before labeling video"
        ) from error

    logger.info("Parsing displayed body parts str")
    try:
        if displayed_body_parts == "all":
            _displayed_body_parts = displayed_body_parts
        else:
            _displayed_body_parts = displayed_body_parts.split(",")
            config = dlc.utils.auxiliaryfunctions.read_config(config_file)
            if any(
                [
                    part not in config["bodyparts"]
                    for part in _displayed_body_parts
                ]
            ):
                raise IdeasError(
                    f"Displayed body parts contains an invalid body part name. Body parts of model are: {config['bodyparts']}"
                )

            logger.info(
                f"Displaying the following body parts: {_displayed_body_parts}"
            )
    except Exception as error:
        raise IdeasError(
            f"Failed to parse displayed body parts ({displayed_body_parts}). Expected format is the value 'all' or a comma-seperated list of body parts tracked by the model."
        ) from error

    try:
        start = time.time()
        dlc.create_labeled_video(
            config_file,
            movie_files,
            destfolder=output_dir,
            filtered=is_filtered,
            displayedbodyparts=displayed_body_parts,
            keypoints_only=keypoints_only,
            outputframerate=output_frame_rate,
            draw_skeleton=draw_skeleton,
            trailpoints=trail_points,
            displaycropped=True,
            codec="avc1",
        )

        if not (
            [
                file
                for file in os.listdir(output_dir)
                if file.lower().endswith("_labeled.mp4")
            ]
        ):
            raise IdeasError("No labeled mp4 output generated by DeepLabCut.")

        end = time.time()
        logger.info(
            (
                "Finished labeling DeepLabCut pose estimates with "
                f"overall performance of {num_frames / (end - start)} FPS"
            )
        )
    except Exception as error:
        raise IdeasError("Failed to label DeepLabCut pose estimates")
