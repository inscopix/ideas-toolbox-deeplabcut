import os
import zipfile

import pytest

from toolbox.tools.training import train_model

input_dir = "/ideas/data/training"


@pytest.mark.parametrize(
    "labeled_data_zip_file,config_file,train_pose_cfg_file,test_pose_cfg_file,max_iters,save_iters,plot_evaluation_results,per_keypoint_evaluation,generate_maps",
    [
        pytest.param(
            [os.path.join(input_dir, "labeled-data.zip")],
            [os.path.join(input_dir, "config.yaml")],
            [os.path.join(input_dir, "train_pose_cfg.yaml")],
            [os.path.join(input_dir, "test_pose_cfg.yaml")],
            2000,
            1000,
            True,
            True,
            True,
            marks=pytest.mark.skipif(
                not int(os.getenv("USE_GPU", default=0)),
                reason="Test is only for GPUs",
            ),
        )
    ],
)
def test_train_model(
    labeled_data_zip_file,
    config_file,
    train_pose_cfg_file,
    test_pose_cfg_file,
    max_iters,
    save_iters,
    plot_evaluation_results,
    per_keypoint_evaluation,
    generate_maps,
    output_dir,
):
    """Tests basic functionality of the train model tool"""

    train_model(
        labeled_data_zip_file=labeled_data_zip_file,
        config_file=config_file,
        train_pose_cfg_file=train_pose_cfg_file,
        test_pose_cfg_file=test_pose_cfg_file,
        max_iters=max_iters,
        save_iters=save_iters,
        plot_evaluation_results=plot_evaluation_results,
        per_keypoint_evaluation=per_keypoint_evaluation,
        generate_maps=generate_maps,
        output_dir=output_dir,
        engine="tensorflow",
    )

    dlc_models_zip = os.path.join(output_dir, "model.zip")
    assert os.path.exists(dlc_models_zip)

    # verify contents of dlc-models folder
    with zipfile.ZipFile(dlc_models_zip, "r") as f:
        f.extractall(output_dir)

    assert set(
        ["config.yaml", "CombinedEvaluation-results.csv", "dlc-models"]
    ).issubset(os.listdir(output_dir))
    assert any(
        [f.endswith("_error_distribution.h5") for f in os.listdir(output_dir)]
    )
    assert any(
        [
            f.endswith("_error_distribution_pcutoff.h5")
            for f in os.listdir(output_dir)
        ]
    )
    assert "iteration-0" in os.listdir(os.path.join(output_dir, "dlc-models"))
    assert os.listdir(
        os.path.join(output_dir, "dlc-models", "iteration-0")
    ) == ["bottom-view-mouseJul16-trainset95shuffle1"]
    assert set(
        os.listdir(
            os.path.join(
                output_dir,
                "dlc-models",
                "iteration-0",
                "bottom-view-mouseJul16-trainset95shuffle1",
            )
        )
    ) == set(["test", "train"])
    assert set(
        os.listdir(
            os.path.join(
                output_dir,
                "dlc-models",
                "iteration-0",
                "bottom-view-mouseJul16-trainset95shuffle1",
                "train",
            )
        )
    ) == set(
        [
            "checkpoint",
            "learning_stats.csv",
            "log",
            "log.txt",
            "pose_cfg.yaml",
            "snapshot-2000.data-00000-of-00001",
            "snapshot-2000.index",
            "snapshot-2000.meta",
        ]
    )
    assert os.listdir(
        os.path.join(
            output_dir,
            "dlc-models",
            "iteration-0",
            "bottom-view-mouseJul16-trainset95shuffle1",
            "test",
        )
    ) == ["pose_cfg.yaml"]

    evaluation_results_zip = os.path.join(output_dir, "evaluation_results.zip")
    assert os.path.exists(evaluation_results_zip)

    # verify contents of evaluation-results folder
    with zipfile.ZipFile(evaluation_results_zip, "r") as f:
        f.extractall(output_dir)

    assert "evaluation-results" in os.listdir(output_dir)
    assert os.listdir(os.path.join(output_dir, "evaluation-results")) == [
        "iteration-0"
    ]
    assert set(
        os.listdir(
            os.path.join(output_dir, "evaluation-results", "iteration-0")
        )
    ) == set(
        [
            "bottom-view-mouseJul16-trainset95shuffle1",
            "CombinedEvaluation-results.csv",
            "DLC_resnet50_bottom-view-mouseJul16shuffle1_2000_error_distribution.h5",
            "DLC_resnet50_bottom-view-mouseJul16shuffle1_2000_error_distribution_pcutoff.h5",
        ]
    )
    assert set(
        os.listdir(
            os.path.join(
                output_dir,
                "evaluation-results",
                "iteration-0",
                "bottom-view-mouseJul16-trainset95shuffle1",
            )
        )
    ) == set(
        [
            "DLC_resnet50_bottom-view-mouseJul16shuffle1_2000-keypoint-results.csv",
            "DLC_resnet50_bottom-view-mouseJul16shuffle1_2000-results.csv",
            "DLC_resnet50_bottom-view-mouseJul16shuffle1_2000-snapshot-2000.h5",
            "LabeledImages_DLC_resnet50_bottom-view-mouseJul16shuffle1_2000_snapshot-2000",
            "DLC_resnet50_bottom-view-mouseJul16shuffle1_1000-keypoint-results.csv",
            "DLC_resnet50_bottom-view-mouseJul16shuffle1_1000-snapshot-1000.h5",
            "LabeledImages_DLC_resnet50_bottom-view-mouseJul16shuffle1_1000_snapshot-1000",
            "maps",
        ]
    )

    # verify previews
    preview_files = [
        "evaluation_results_preview.svg",
        "evaluation_results_error_distribution_preview.svg",
        "evaluation_results_error_distribution_pcutoff_preview.svg",
        "evaluation_results_plots_train_movie_preview.mp4",
        "evaluation_results_plots_test_movie_preview.mp4",
        "evaluation_results_maps_locref_movie_preview.mp4",
        "evaluation_results_maps_scmap_movie_preview.mp4",
        "learning_stats_preview.svg",
    ]

    for f in preview_files:
        assert os.path.exists(os.path.join(output_dir, f))


@pytest.mark.parametrize(
    "labeled_data_zip_file,config_file,train_pose_cfg_file,test_pose_cfg_file,max_iters,save_iters,plot_evaluation_results,per_keypoint_evaluation,generate_maps",
    [
        pytest.param(
            [os.path.join(input_dir, "labeled-data.zip")],
            [os.path.join(input_dir, "config.yaml")],
            [os.path.join(input_dir, "train_pose_cfg.yaml")],
            [os.path.join(input_dir, "test_pose_cfg.yaml")],
            2000,
            1000,
            True,
            True,
            True,
            marks=pytest.mark.skipif(
                not int(os.getenv("USE_GPU", default=0)),
                reason="Test is only for GPUs",
            ),
        )
    ],
)
def test_train_model_pytorch(
    labeled_data_zip_file,
    config_file,
    train_pose_cfg_file,
    test_pose_cfg_file,
    max_iters,
    save_iters,
    plot_evaluation_results,
    per_keypoint_evaluation,
    generate_maps,
    output_dir,
):
    """Tests basic functionality of the train model tool using pytorch engine"""

    train_model(
        labeled_data_zip_file=labeled_data_zip_file,
        config_file=config_file,
        train_pose_cfg_file=train_pose_cfg_file,
        test_pose_cfg_file=test_pose_cfg_file,
        max_iters=max_iters,
        save_iters=save_iters,
        epochs=3,
        save_epochs=1,
        plot_evaluation_results=plot_evaluation_results,
        per_keypoint_evaluation=per_keypoint_evaluation,
        generate_maps=generate_maps,
        output_dir=output_dir,
        engine="pytorch",
    )

    dlc_models_zip = os.path.join(output_dir, "model.zip")
    assert os.path.exists(dlc_models_zip)

    # verify contents of dlc-models folder
    with zipfile.ZipFile(dlc_models_zip, "r") as f:
        f.extractall(output_dir)

    # ['evaluation_results_preview.svg', 'evaluation_results_error_distribution_preview.svg', 'DLC_Resnet50_bottom-view-mouseJul16shuffle1_snapshot_003_error_distribution_pcutoff.h5', 'evaluation_results_error_distribution_pcutoff_preview.svg', 'model.zip', 'evaluation_results.zip', 'learning_stats_preview.svg', 'config.yaml', 'evaluation_results_plots_test_movie_preview.mp4', 'DLC_Resnet50_bottom-view-mouseJul16shuffle1_snapshot_003_error_distribution.h5', 'evaluation_results_plots_train_movie_preview.mp4', 'dlc-models-pytorch', 'CombinedEvaluation-results.csv']
    assert set(
        ["config.yaml", "CombinedEvaluation-results.csv", "dlc-models-pytorch"]
    ).issubset(os.listdir(output_dir))
    assert any(
        [f.endswith("_error_distribution.h5") for f in os.listdir(output_dir)]
    )
    assert any(
        [
            f.endswith("_error_distribution_pcutoff.h5")
            for f in os.listdir(output_dir)
        ]
    )
    assert "iteration-0" in os.listdir(os.path.join(output_dir, "dlc-models-pytorch"))
    assert os.listdir(
        os.path.join(output_dir, "dlc-models-pytorch", "iteration-0")
    ) == ["bottom-view-mouseJul16-trainset95shuffle1"]
    assert set(
        os.listdir(
            os.path.join(
                output_dir,
                "dlc-models-pytorch",
                "iteration-0",
                "bottom-view-mouseJul16-trainset95shuffle1",
            )
        )
    ) == set(["test", "train"])

    assert set(
        os.listdir(
            os.path.join(
                output_dir,
                "dlc-models-pytorch",
                "iteration-0",
                "bottom-view-mouseJul16-trainset95shuffle1",
                "train",
            )
        )
    ) == set(
        [
            "learning_stats.csv",
            "pytorch_config.yaml",
            "snapshot-003.pt",
            "train.txt"
        ]
    )
    assert os.listdir(
        os.path.join(
            output_dir,
            "dlc-models-pytorch",
            "iteration-0",
            "bottom-view-mouseJul16-trainset95shuffle1",
            "test",
        )
    ) == ["pose_cfg.yaml"]

    evaluation_results_zip = os.path.join(output_dir, "evaluation_results.zip")
    assert os.path.exists(evaluation_results_zip)

    # verify contents of evaluation-results folder
    with zipfile.ZipFile(evaluation_results_zip, "r") as f:
        f.extractall(output_dir)
    print("HIIIII", os.listdir(output_dir))

    assert "evaluation-results-pytorch" in os.listdir(output_dir)
    assert os.listdir(os.path.join(output_dir, "evaluation-results-pytorch")) == [
        "iteration-0"
    ]
    assert set(
        os.listdir(
            os.path.join(output_dir, "evaluation-results-pytorch", "iteration-0")
        )
    ) == set(
        [
            "bottom-view-mouseJul16-trainset95shuffle1",
            "CombinedEvaluation-results.csv",
            "DLC_Resnet50_bottom-view-mouseJul16shuffle1_snapshot_003_error_distribution.h5",
            "DLC_Resnet50_bottom-view-mouseJul16shuffle1_snapshot_003_error_distribution_pcutoff.h5",
        ]
    )
    assert set(
        os.listdir(
            os.path.join(
                output_dir,
                "evaluation-results-pytorch",
                "iteration-0",
                "bottom-view-mouseJul16-trainset95shuffle1",
            )
        )
    ) == set(
        [
            "DLC_Resnet50_bottom-view-mouseJul16shuffle1_snapshot_003-results.csv",
            "DLC_Resnet50_bottom-view-mouseJul16shuffle1_snapshot_003-keypoint-results.csv",
            "DLC_Resnet50_bottom-view-mouseJul16shuffle1_snapshot_003.h5",
            "LabeledImages_DLC_Resnet50_bottom-view-mouseJul16shuffle1_snapshot_003",
            "DLC_Resnet50_bottom-view-mouseJul16shuffle1_snapshot_002-results.csv",
            "DLC_Resnet50_bottom-view-mouseJul16shuffle1_snapshot_002-keypoint-results.csv",
            "DLC_Resnet50_bottom-view-mouseJul16shuffle1_snapshot_002.h5",
            "LabeledImages_DLC_Resnet50_bottom-view-mouseJul16shuffle1_snapshot_002",
            "DLC_Resnet50_bottom-view-mouseJul16shuffle1_snapshot_001-results.csv",
            "DLC_Resnet50_bottom-view-mouseJul16shuffle1_snapshot_001-keypoint-results.csv",
            "DLC_Resnet50_bottom-view-mouseJul16shuffle1_snapshot_001.h5",
            "LabeledImages_DLC_Resnet50_bottom-view-mouseJul16shuffle1_snapshot_001",
        ]
    )

    # verify previews
    preview_files = [
        "evaluation_results_preview.svg",
        "evaluation_results_error_distribution_preview.svg",
        "evaluation_results_error_distribution_pcutoff_preview.svg",
        "evaluation_results_plots_train_movie_preview.mp4",
        "evaluation_results_plots_test_movie_preview.mp4",
        "learning_stats_preview.svg",
    ]

    for f in preview_files:
        assert os.path.exists(os.path.join(output_dir, f))
