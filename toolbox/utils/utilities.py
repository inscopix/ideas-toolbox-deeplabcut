import os
import cv2
import numpy as np
import imageio
from deeplabcut.utils.auxiliaryfunctions import read_config, write_config
from pathlib import Path
from ideas.exceptions import IdeasError
import isx


def compute_sampling_rate(period_num: int, period_den: int) -> float:
    """Compute the sampling rate given the period numerator and denominator.

    :param period_num: numerator in the period
    :param period_den: denominator in the period
    :return: the sampling rate or None if there is a division by zero error.
    """
    try:
        return np.round(1 / (period_num / period_den), 2)
    except ZeroDivisionError:
        return None


def count_video_frames(input_video_filename, method="manual"):
    """Count the number of frames in a video recording using the specified approach.

    :param input_video_filename: path to the input video file
    :param method: string specifying the method to use ('manual' or 'imageio')
    """
    if method == "manual":
        # The manual method uses opencv to read every frame and increment a counter
        cap = cv2.VideoCapture(input_video_filename)
        num_frames = 0
        while True:
            status, frame = cap.read()
            if not status:
                break
            num_frames += 1
        cap.release()
    elif method == "imageio":
        # The imageio method retrieves number of frames from mp4 metadata.
        # This method is faster since it does not require reading all frames
        video_reader = imageio.get_reader(input_video_filename, "ffmpeg")
        num_frames = video_reader.count_frames()
    return num_frames


def get_timing_spacing_metadata_from_labeled_movie(movie_file: str):
    """Get timing and spacing metadata from labeled movie.
    Attach to the IDEAS output metadata.

    :param str movie_file: Path the labeled movie file.
    """
    movie = cv2.VideoCapture(movie_file)
    sampling_rate = movie.get(cv2.CAP_PROP_FPS)
    num_frames = count_video_frames(movie_file, method="imageio")

    return {
        "spacingInfo": {
            "numPixels": {
                "x": int(movie.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "y": int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
        },
        "timingInfo": {
            "cropped": [],
            "dropped": [],
            "numTimes": num_frames,
            "period": {"num": int(1000000 / sampling_rate), "den": 1000000},
            "sampling_rate": compute_sampling_rate(
                period_num=int(1000000 / sampling_rate), period_den=1000000
            ),
        },
    }


def update_config_file(config_file: str, updates: dict):
    """Update the config.yaml file in the pre-trained model directory.
    Currently this is the only way to control certain parameters in the DeepLabCut workflow.

    :param str config_file: Path the config.yaml file.
    :param dict updates: Dictionary of updates to apply on the config.yaml file.
    :raises KeyError: If a key in the updates dict does not exist already in the config.yaml file.
    """
    config = read_config(config_file)
    for key, value in updates.items():
        if key not in config:
            raise KeyError(
                f"No key ({key}) to update in config file ({config_file})."
            )
        config[key] = value
    write_config(config_file, config)


def _order_isxb_movie_files(isxb_movie_files):
    """Order isxb movie files based on their start time.

    :param str movie_files: List of movie file paths
    """
    if len(isxb_movie_files) < 2:
        return isxb_movie_files

    start_times = np.zeros(len(isxb_movie_files))
    pixel_shapes = [None] * len(isxb_movie_files)

    for i, f in enumerate(isxb_movie_files):
        # ensure file exists
        if not os.path.exists(f):
            raise IdeasError(
                "The file '{0}' could not be found".format(os.path.basename(f))
            )

        # ensure file extension is '.isxb'
        _, file_extension = os.path.splitext(os.path.basename(f))
        file_extension = file_extension.lower()
        if file_extension != ".isxb":
            raise IdeasError(
                "File extension {0} is not supported, only isxb files are supported".format(
                    file_extension
                )
            )

        # read the metadata and ensure that all the pixel shapes are the same
        movie = isx.Movie.read(f)
        pixel_shapes[i] = movie.spacing.num_pixels

        # read start time of the movie
        start_times[i] = movie.timing.start.to_datetime().timestamp()

    if pixel_shapes.count(pixel_shapes[0]) != len(pixel_shapes):
        raise IdeasError(
            "The input files do not form a series. The pixel sizes of the files provided do not match."
        )

    ordered_indices = np.argsort(start_times)
    ordered_isxb_movie_files = np.array(isxb_movie_files)[
        ordered_indices
    ].tolist()
    return ordered_isxb_movie_files


def order_movie_files(movie_files):
    """Order movie files based on their start time.

    :param str movie_files: List of movie file paths
    """
    if len(movie_files) < 2:
        return movie_files

    file_ext = Path(movie_files[0]).suffix[1:].lower()
    if file_ext == "isxb":
        return _order_isxb_movie_files(movie_files)
    else:
        return movie_files
