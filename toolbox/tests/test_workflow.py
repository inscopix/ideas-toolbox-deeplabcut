import os
import json

import pytest
import pandas as pd
import numpy as np

from toolbox.tools.workflow import run_workflow
from toolbox.utils.testing_utils import assert_mp4_movies_are_close
from toolbox.utils.io import read_annotations

input_dir = "/ideas/data/user_models"


@pytest.mark.parametrize(
    "model_dir,movie_files,experiment_annotations_format,crop_rect,window_length,displayed_body_parts,p_cutoff,dot_size,color_map,keypoints_only,output_frame_rate,draw_skeleton,trail_points",
    [
        pytest.param(
            [
                f"{input_dir}/saviochan_Deeplabcut-OpenFieldArena-IDEAS-2024-04-01.zip"
            ],
            [f"{input_dir}/m3v1mp4_trimmed_1s.mp4"],
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
    ],
)
def test_run_workflow(
    model_dir,
    movie_files,
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
    """Tests the workflow tool on a zipped model dir."""
    dataset_name, _ = os.path.splitext(os.path.basename(movie_files[0]))

    run_workflow(
        model_dir=model_dir,
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

    # assert (df.columns == expected_df.columns).all()
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
