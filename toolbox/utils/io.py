import os
import pandas as pd
import numpy as np
import isx
import cv2
from ideas.exceptions import IdeasError
from toolbox.utils.utilities import count_video_frames
from toolbox.utils import config


def get_timestamps_from_movies(movie_files):
    """Get timestamps from a movie.
    Timestamps can represent either:
        1. The time since the start of the movie in seconds
            This type of timestamps will be output for all valid movie types.
        2. The raw hardware counter value read from the .isxb file.
            This can be used for synchronization with .isxd data in downstream analysis.
            If the movie is not .isxb, then these types of timestamps are not included in the output.

    :param movie_files: The movie files to read from. Valid file types are: .isxb, .mp4, .avi.

    :return data_dict: A dictionary containing a map of different types of timestamps.
        The keys in the dictionary are the labels describing the type of timestamps.
        The values in the dictionary are the timestamps.
    """
    cumulative_num_frames = 0
    cumulative_start_time = 0
    data = {
        "Frame number": np.array([], dtype="int64"),
        "Movie number": np.array([], dtype="int64"),
        "Local frame number": np.array([], dtype="int64"),
        "Time since start (s)": np.array([], dtype="float64"),
    }

    for movie_index, movie_file in enumerate(movie_files):
        if movie_file.lower().endswith(".isxb"):
            if "Hardware counter (us)" not in data:
                data["Hardware counter (us)"] = np.array([], dtype="float64")

            movie = isx.Movie.read(movie_file)
            frame_number = np.arange(movie.timing.num_samples)
            ts_start = np.empty(shape=(movie.timing.num_samples,))
            ts_raw = np.empty(shape=(movie.timing.num_samples,))
            period = movie.timing.period.secs_float
            start_tsc = None

            for i in range(movie.timing.num_samples):
                ts_raw[i] = movie.get_frame_timestamp(i)

                if start_tsc is None:
                    # first movie in series, grab tsc value to compute offsets
                    start_tsc = ts_raw[0]

                # directly compute offset to get time since start, this works because the series files are synchronized
                ts_start[i] = (ts_raw[i] - start_tsc) / 1e6

            current_frame_number = cumulative_num_frames + frame_number
            current_movie_number = [movie_index] * len(frame_number)
            current_local_frame_number = frame_number
            current_ts_start = cumulative_start_time + ts_start
            current_ts_raw = ts_raw

            data["Hardware counter (us)"] = np.concatenate(
                (data["Hardware counter (us)"], current_ts_raw)
            )

        elif movie_file.lower().endswith(
            ".mp4"
        ) or movie_file.lower().endswith(".avi"):
            movie = cv2.VideoCapture(movie_file)
            sampling_rate = movie.get(cv2.CAP_PROP_FPS)
            period = 1 / sampling_rate

            if sampling_rate == 0:
                raise IdeasError(
                    f"Cannot read sampling rate from ({movie_file})."
                )

            period = 1 / sampling_rate
            num_frames = count_video_frames(movie_file, method="imageio")
            frame_number = np.arange(num_frames)
            ts_start = np.empty(shape=(num_frames,))

            for i in range(num_frames):
                ts_start[i] = i * period

            current_frame_number = cumulative_num_frames + frame_number
            current_movie_number = [movie_index] * len(frame_number)
            current_local_frame_number = frame_number
            current_ts_start = cumulative_start_time + ts_start
        else:
            raise IdeasError(
                f"Cannot get timestamps from movie file ({os.path.basename(movie_file)}) with unsupported file extension."
            )

        data["Frame number"] = np.concatenate(
            (data["Frame number"], current_frame_number)
        )
        data["Movie number"] = np.concatenate(
            (data["Movie number"], current_movie_number)
        )
        data["Local frame number"] = np.concatenate(
            (data["Local frame number"], current_local_frame_number)
        )
        data["Time since start (s)"] = np.concatenate(
            (data["Time since start (s)"], current_ts_start)
        )

        cumulative_num_frames += len(frame_number)
        cumulative_start_time = data["Time since start (s)"][-1] + period

    return data


def read_annotations(annotations_file: str) -> pd.DataFrame:
    """Read annotations from disk to memory.

    :param annotations_file str: The annotations file to read from.
        Valid file extensions are: .parquet, .csv
    """

    ext = os.path.splitext(annotations_file)[1].lower()
    if ext not in config.ANNOTATIONS_FILE_EXTS:
        raise IdeasError(
            (
                f"Unknown file extension ({ext}) for annotations file ({annotations_file}). "
                f"Must be one of the following formats: {config.ANNOTATIONS_FILE_EXTS}"
            ),
        )

    if ext == config.PARQUET_FILE_EXT:
        return pd.read_parquet(annotations_file)
    elif ext == config.CSV_FILE_EXT:
        return pd.read_csv(annotations_file)


def write_annotations(annotations_df: pd.DataFrame, annotations_file: str):
    """Writes annotations from memory to disk.

    :param annotations_df pd.DataFrame: The data to write to disk.
    :param annotations_file str: The annotations file to write to.
        Valid file extensions are: .parquet, .csv
    """

    ext = os.path.splitext(annotations_file)[1].lower()
    if ext not in config.ANNOTATIONS_FILE_EXTS:
        raise IdeasError(
            (
                f"Unknown file extension ({ext}) for annotations file ({annotations_file}). "
                f"Must be one of the following formats: {config.ANNOTATIONS_FILE_EXTS}"
            ),
        )

    if ext == config.PARQUET_FILE_EXT:
        annotations_df.to_parquet(annotations_file)
    elif ext == config.CSV_FILE_EXT:
        annotations_df.to_csv(annotations_file, index=False)


def convert_dlc_h5_to_annotations(
    pose_estimates_h5_files, movie_files, annotations_file
):
    """Convert the DeepLabCut .h5 output file to a .parquet/.csv output experiemnt annotations file.

    The annotations file will contain one more columns containing the timestamps for each sample,
    and three columns for each body part containing the x, y, and likelihood values for the body part.

    :param pose_estimates_h5_files: The .h5 file output by DeepLabCut containing the pose estimates in a multi-index pandas array.
    :param movie_files: List of movie files to read timestamps from. This should be the movie file that generated the pose estimates .h5 file.
        Valid file types are: .isxb, .mp4, .avi.
    :param annotations_file: The output file to store the experiment annotations.
        Valid file types are: .parquet, .csv.
    """
    dfs = [
        pd.read_hdf(file).droplevel("scorer", axis=1)
        for file in pose_estimates_h5_files
    ]
    data_dict = get_timestamps_from_movies(movie_files)
    body_parts = dfs[0].columns.get_level_values("bodyparts").unique().tolist()

    for body_part in body_parts:
        data_dict[f"{body_part} x"] = []
        data_dict[f"{body_part} y"] = []
        data_dict[f"{body_part} likelihood"] = []

        for df in dfs:
            data_dict[f"{body_part} x"] += df[body_part]["x"].tolist()
            data_dict[f"{body_part} y"] += df[body_part]["y"].tolist()
            data_dict[f"{body_part} likelihood"] += df[body_part][
                "likelihood"
            ].tolist()

    new_df = pd.DataFrame(data_dict)
    write_annotations(new_df, annotations_file)
