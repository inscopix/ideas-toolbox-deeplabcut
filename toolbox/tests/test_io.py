import os

import pytest
import numpy as np
import pandas as pd

from ideas.exceptions import IdeasError

from toolbox.utils.io import (
    get_timestamps_from_movies,
    convert_dlc_h5_to_annotations,
)

input_dir = "/ideas/data"
bottom_model_name = "_dlc_pose_estimates"

TEST_DATASET_AVI = (
    "2023-01-31-12-12-08_video_e3v8251-20230131T121212-122254_trimmed_1s"
)
TEST_MOVIE_FILE_AVI = os.path.join(input_dir, TEST_DATASET_AVI + ".avi")

TEST_DATASET_ISXB = "2023-01-27-10-34-22-camera-1_trimmed_1s"
TEST_MOVIE_FILE_ISXB = os.path.join(input_dir, TEST_DATASET_ISXB + ".isxb")
TEST_POSE_ESTIMATES_FILE_ISXB = os.path.join(
    input_dir, TEST_DATASET_ISXB + bottom_model_name + ".h5"
)


@pytest.mark.parametrize(
    ("movie_file", "expected"),
    [
        (
            TEST_MOVIE_FILE_ISXB,
            {
                "Frame number": [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                ],
                "Movie number": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                "Local frame number": [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                ],
                "Time since start (s)": [
                    0.0,
                    0.052012,
                    0.100012,
                    0.15201,
                    0.200012,
                    0.251999,
                    0.300013,
                    0.352026,
                    0.404023,
                    0.452017,
                    0.504027,
                    0.552005,
                    0.604001,
                    0.65201,
                    0.703997,
                    0.752002,
                    0.804005,
                    0.852022,
                    0.904023,
                    0.952024,
                ],
                "Hardware counter (us)": [
                    247903537725,
                    247903589737,
                    247903637737,
                    247903689735,
                    247903737737,
                    247903789724,
                    247903837738,
                    247903889751,
                    247903941748,
                    247903989742,
                    247904041752,
                    247904089730,
                    247904141726,
                    247904189735,
                    247904241722,
                    247904289727,
                    247904341730,
                    247904389747,
                    247904441748,
                    247904489749,
                ],
            },
        ),
        (
            TEST_MOVIE_FILE_AVI,
            {
                "Frame number": [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                ],
                "Movie number": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                "Local frame number": [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                ],
                "Time since start (s)": [
                    0.0,
                    0.05,
                    0.1,
                    0.15,
                    0.2,
                    0.25,
                    0.3,
                    0.35,
                    0.4,
                    0.45,
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                ],
            },
        ),
    ],
)
def test_get_timestamps_from_movies(movie_file, expected):
    """
    Test timestamps are read and formatted from valid movie files correctly.
    """
    actual = get_timestamps_from_movies([movie_file])
    assert sorted(actual.keys()) == sorted(expected.keys())
    for key in actual.keys():
        np.testing.assert_allclose(actual[key], expected[key])


def test_get_timestamps_from_movies_series():
    """
    Test timestamps are read and formatted from valid movie series correctly.
    """
    movie_files = [
        "/ideas/data/series/Group-20240111-080531_2024-01-12-08-55-21_sched_0.isxb",
        "/ideas/data/series/Group-20240111-080531_2024-01-12-08-55-21_sched_1.isxb",
    ]

    expected = {
        "Frame number": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
        ],
        "Movie number": [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        "Local frame number": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
        ],
        "Time since start (s)": [
            0.0,
            0.031986,
            0.063986,
            0.099984,
            0.132004,
            0.164004,
            0.199985,
            0.231983,
            0.263992,
            0.299985,
            0.33332831,
            0.36533531,
            0.40132731,
            0.43333031,
            0.46533631,
            0.50132131,
            0.53332831,
            0.56534831,
            0.60136631,
            0.63332731,
        ],
        "Hardware counter (us)": [
            163957519943,
            163957551929,
            163957583929,
            163957619927,
            163957651947,
            163957683947,
            163957719928,
            163957751926,
            163957783935,
            163957819928,
            163957851928,
            163957883935,
            163957919927,
            163957951930,
            163957983936,
            163958019921,
            163958051928,
            163958083948,
            163958119966,
            163958151927,
        ],
    }

    actual = get_timestamps_from_movies(movie_files)
    assert sorted(actual.keys()) == sorted(expected.keys())
    for key in actual.keys():
        np.testing.assert_allclose(actual[key], expected[key])


@pytest.mark.parametrize(
    "movie_file", ["/path/to/movie.isxd", "/path/to/movie.mov"]
)
def test_get_timestamps_from_movies_invalid_file_extension(movie_file):
    """
    Test timestamps cannot be read from invalid movie files.
    """

    with pytest.raises(IdeasError) as cm:
        get_timestamps_from_movies([movie_file])

    exception = cm.value
    assert str(exception) == (
        f"Cannot get timestamps from movie file ({os.path.basename(movie_file)}) with unsupported file extension."
    )


@pytest.mark.parametrize(
    (
        "pose_estimates_h5_file",
        "movie_file",
        "expected_columns",
        "expected_first_row",
        "expected_last_row",
    ),
    [
        (
            TEST_POSE_ESTIMATES_FILE_ISXB,
            TEST_MOVIE_FILE_ISXB,
            [
                "Frame number",
                "Movie number",
                "Local frame number",
                "Time since start (s)",
                "Hardware counter (us)",
                "tail_tip x",
                "tail_tip y",
                "tail_tip likelihood",
                "tail_base x",
                "tail_base y",
                "tail_base likelihood",
                "R_hind x",
                "R_hind y",
                "R_hind likelihood",
                "L_hind x",
                "L_hind y",
                "L_hind likelihood",
                "neck x",
                "neck y",
                "neck likelihood",
                "R_fore x",
                "R_fore y",
                "R_fore likelihood",
                "L_fore x",
                "L_fore y",
                "L_fore likelihood",
                "nose x",
                "nose y",
                "nose likelihood",
            ],
            [
                0.0,
                0,
                0,
                0.0,
                247903537725.0,
                564.8240966796875,
                317.3860778808594,
                0.9447007775306702,
                509.9263000488281,
                351.196533203125,
                0.9994614124298096,
                528.37744140625,
                390.61273193359375,
                0.9971207976341248,
                476.92999267578125,
                352.4303283691406,
                0.977459728717804,
                491.11468505859375,
                454.84625244140625,
                0.9713097810745239,
                492.220703125,
                422.94659423828125,
                0.9939277768135071,
                467.02880859375,
                440.4466247558594,
                0.9846889972686768,
                488.832763671875,
                484.03619384765625,
                0.9942392706871033,
            ],
            [
                19.0,
                0,
                19,
                0.952024,
                247904489749.0,
                298.591064453125,
                330.63983154296875,
                0.8848316073417664,
                476.463134765625,
                391.84228515625,
                0.9990378022193909,
                511.1584167480469,
                432.4688415527344,
                0.9953562021255493,
                444.1005859375,
                409.6938781738281,
                0.9980884790420532,
                460.1864929199219,
                514.400390625,
                0.9760664105415344,
                479.8973083496094,
                505.99267578125,
                0.9978686571121216,
                456.5481262207031,
                471.1663513183594,
                0.9417847990989685,
                449.5748291015625,
                545.3216552734375,
                0.98087078332901,
            ],
        ),
    ],
)
def test_convert_dlc_h5_to_annotations(
    output_dir,
    pose_estimates_h5_file,
    movie_file,
    expected_columns,
    expected_first_row,
    expected_last_row,
):
    """
    Test the DeepLabCut .h5 file output is converted to experiment annotations correctly.

    This function tests conversion to both parquet and csv yields the same expected data.
    The first and last rows of the annotations are compared with the actual output due to the large output size.
    """
    extensions = [".parquet", ".csv"]
    for extension in extensions:
        annotations_file_parquet = os.path.join(
            output_dir, f"annotations{extension}"
        )
        convert_dlc_h5_to_annotations(
            [pose_estimates_h5_file], [movie_file], annotations_file_parquet
        )

        if extension == ".parquet":
            annotations_df = pd.read_parquet(annotations_file_parquet)
        else:
            annotations_df = pd.read_csv(annotations_file_parquet)

        assert list(annotations_df.columns) == expected_columns
        np.testing.assert_allclose(
            annotations_df.iloc[0].tolist(), expected_first_row
        )
        np.testing.assert_allclose(
            annotations_df.iloc[-1].tolist(), expected_last_row
        )
