import os
import json
import pytest

import matplotlib
import pandas as pd
import numpy as np
from ideas.exceptions import IdeasError

from toolbox.tools.inscopix_bottom_up_model import run
from toolbox.utils.testing_utils import assert_mp4_movies_are_close
from toolbox.utils.io import read_annotations

input_dir = "/ideas/data"


@pytest.mark.parametrize(
    "movie_file,experiment_annotations_format,crop_rect,window_length,displayed_body_parts,p_cutoff,dot_size,color_map,keypoints_only,output_frame_rate,draw_skeleton,trail_points",
    [
        pytest.param(
            "2023-01-27-10-34-22-camera-1_trimmed_1s.isxb",
            "csv",
            [],
            5,
            "all",
            0.6,
            5,
            "rainbow",
            False,
            20,
            False,
            0,
            marks=pytest.mark.skipif(
                not int(os.getenv("USE_GPU", default=0)),
                reason="Test is only for GPUs",
            ),
        ),
        pytest.param(
            "2023-01-31-12-12-08_video_e3v8251-20230131T121212-122254_trimmed_1s.avi",
            "parquet",
            [
                {
                    "groupKey": "crop_rect",
                    "left": 325,
                    "top": 125,
                    "width": 1000,
                    "height": 1000,
                    "name": "Crop Rectangle",
                    "stroke": "#EB823B",
                    "type": "boundingBox",
                }
            ],
            0,
            "neck,nose",
            0.4,
            3,
            "summer",
            False,
            None,
            True,
            5,
            marks=pytest.mark.skipif(
                not int(os.getenv("USE_GPU", default=0)),
                reason="Test is only for GPUs",
            ),
        ),
    ],
)
def test_inscopix_bottom_up_model_run(
    movie_file,
    experiment_annotations_format,
    crop_rect,
    window_length,
    displayed_body_parts,
    p_cutoff,
    dot_size,
    color_map,
    keypoints_only,
    output_frame_rate,
    draw_skeleton,
    trail_points,
    output_dir,
):
    """Test the inscopix bottom up model workflow"""

    dataset_name, _ = os.path.splitext(movie_file)
    movie_files = [f"{input_dir}/{movie_file}"]
    run(
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

    # verify h5 output
    h5_output_filename = f"{output_dir}/dlc_pose_estimates.0.h5"
    assert os.path.exists(h5_output_filename)
    df = pd.read_hdf(h5_output_filename)

    expected_h5_output_filename = (
        f"{input_dir}/{dataset_name}_dlc_pose_estimates.h5"
    )
    expected_df = pd.read_hdf(expected_h5_output_filename)

    assert (df.columns == expected_df.columns).all()
    np.testing.assert_allclose(
        df.to_numpy(), expected_df.to_numpy(), rtol=1e-1
    )

    # verify csv output
    annotations_output_filename = (
        f"{output_dir}/dlc_annotations.{experiment_annotations_format}"
    )
    assert os.path.exists(annotations_output_filename)
    df = read_annotations(annotations_output_filename)

    expected_annotations_output_filename = f"{input_dir}/{dataset_name}_dlc_annotations.{experiment_annotations_format}"
    expected_df = read_annotations(expected_annotations_output_filename)

    assert (df.columns == expected_df.columns).all()
    np.testing.assert_allclose(
        df.to_numpy(), expected_df.to_numpy(), rtol=1e-1
    )

    # verify mp4 output
    mp4_output_filename = f"{output_dir}/dlc_labeled_movie.0.mp4"
    assert os.path.exists(mp4_output_filename)

    expected_mp4_output_filename = (
        f"{input_dir}/{dataset_name}_dlc_labeled_movie.mp4"
    )
    assert_mp4_movies_are_close(
        mp4_output_filename, expected_mp4_output_filename, tol=1e-1
    )

    # verify output metadata
    with open("output_metadata.json", "r") as f:
        output_metadata = json.load(f)

    with open(f"{input_dir}/{dataset_name}_output_metadata.json", "r") as f:
        expected_output_metadata = json.load(f)

    assert output_metadata == expected_output_metadata


@pytest.mark.parametrize(
    "crop_rect",
    [
        [{"groupKey": "not_crop_rect"}],
        [
            {
                "groupKey": "crop_rect",
            }
        ],
        [{"groupKey": "crop_rect", "top": 0}],
        [{"groupKey": "crop_rect", "top": 0, "left": 0}],
        [{"groupKey": "crop_rect", "top": 0, "left": 0, "width": 0}],
    ],
)
def test_inscopix_bottom_up_model_run_invalid_crop_rect(crop_rect, output_dir):
    """Test the inscopix bottom up model workflow fails with an invalid crop rect"""

    with pytest.raises(IdeasError) as error:
        run(
            movie_files=[
                "/ideas/data/2023-01-27-10-34-22-camera-1_trimmed_1s.isxb"
            ],
            crop_rect=crop_rect,
            output_dir=output_dir,
        )

    assert (
        str(error.value)
        == f"Failed to parse crop rect ({crop_rect}), unexpected format for ROI input."
    )


@pytest.mark.parametrize("displayed_body_parts", ["something", "neck,n0se"])
def test_inscopix_bottom_up_model_run_invalid_displayed_body_parts(
    displayed_body_parts, output_dir
):
    """Test the inscopix bottom up model workflow fails with an invalid displayed body parts"""

    with pytest.raises(IdeasError) as error:
        run(
            movie_files=[
                "/ideas/data/2023-01-27-10-34-22-camera-1_trimmed_1s.isxb"
            ],
            displayed_body_parts=displayed_body_parts,
            output_dir=output_dir,
        )
    assert (
        str(error.value)
        == f"Failed to parse displayed body parts ({displayed_body_parts}). Expected format is the value 'all' or a comma-seperated list of body parts tracked by the model."
    )


@pytest.mark.parametrize(
    "color_map",
    [
        "something",
    ],
)
def test_inscopix_bottom_up_model_run_invalid_color_map(color_map, output_dir):
    """Test the inscopix bottom up model workflow fails with an invalid color map"""

    with pytest.raises(IdeasError) as error:
        run(
            movie_files=[
                "/ideas/data/2023-01-27-10-34-22-camera-1_trimmed_1s.isxb"
            ],
            color_map=color_map,
            output_dir=output_dir,
        )
    assert (
        str(error.value)
        == f"Color map ({color_map}) must be one of the following matplotlib color maps: {matplotlib.colormaps()}"
    )
