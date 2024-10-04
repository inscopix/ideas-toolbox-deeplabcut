import os
import logging
from typing import List, Optional

from toolbox.tools import workflow

logger = logging.getLogger()

MODEL_DIR = "/ideas/models/General_DLC-Ludo-2022-12-16"


def run(
    *,
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
    """Run a DeepLabCut workflow on behaviorial movies using the Inscopix Bottom View Mouse Pose Estimation model.

    The workflow consists of three steps:
    1. Analyze videos using the Inscopix Bottom View Mouse Pose Estimation model
    2. Filter results
    3. Label videos with results

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
    :param str output_dir: Path to the output directory.
    """

    logger.info("Running Inscopix bottom-up mouse pose estimation model")

    if not output_dir:
        output_dir = os.getcwd()

    workflow.run_workflow(
        model_dir=[MODEL_DIR],
        movie_files=movie_files,
        experiment_annotations_format=experiment_annotations_format,
        crop_rect=crop_rect,
        window_length=window_length,
        displayed_body_parts=displayed_body_parts,
        p_cutoff=p_cutoff,
        dot_size=dot_size,
        color_map=color_map,
        keypoints_only=keypoints_only,
        output_frame_rate=output_frame_rate,
        draw_skeleton=draw_skeleton,
        trail_points=trail_points,
        output_dir=output_dir,
    )

    logger.info(
        "Finished running Inscopix bottom-up mouse pose estimation model"
    )
