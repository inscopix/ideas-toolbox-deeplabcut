import cv2
from skimage.metrics import structural_similarity as ssim


def assert_mp4_movies_are_close(file1: str, file2: str, tol: float = 1e-2):
    """Assert two mp4 movies are closely similar to each other.

    The similarily of each frame from both movies is calculated and compared to
    a user-defined threshold.
    Similarity between two frames is calculated using the
    skimage.metrics.structural_similarity function.

    :param str file1: Path to the first mp4 movie file.
    :param str file2: Path to the second mp4 movie file.
    :param float tol: The maximum tolerance of difference between two frames.
        Structural similarity index is value between 1.0 and -1.0,
        where 1 indicates perfect similarity.
        The threshold for similarity is calculated as 1.0 - tol.
    """
    threshold = 1.0 - tol
    files = [file1, file2]
    caps = [cv2.VideoCapture(file) for file in files]
    for file, cap in zip(files, caps):
        if not cap.isOpened():
            raise ValueError(f"Failed to open mp4 file ({file}) for reading.")

    num_frames = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    assert (
        len(set(num_frames)) == 1
    ), "Number of frames in mp4 movies does not match."

    for i in range(num_frames[0]):
        frames = [cap.read()[1] for cap in caps]
        for channel in range(3):
            assert (
                ssim(frames[0][:, :, channel], frames[1][:, :, channel])
                >= threshold
            )
