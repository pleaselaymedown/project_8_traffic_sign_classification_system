"""
GTSRB RGB 28×28 CNN 학습 및 Track 분할 비교
===========================================================

목적
----
Basys3에 최종적으로 사용할 10개 클래스를 선정하기 전에,
선택한 후보 클래스들을 28×28 RGB 입력으로 학습합니다.

이번 실험에서 seed를 사용하는 방법
------------------------------------
- SPLIT_SEEDS = [100]
    어떤 Track이 학습/검증/내부 테스트에 들어가는지만 변경합니다.

- TRAINING_SEED = 100
    CNN 초기 가중치, 데이터 셔플, 데이터 증강의 난수 조건은 고정합니다.

즉, 모델 구조나 학습 조건은 동일하게 유지하고,
서로 다른 실제 표지판 Track 조합에서도 각 클래스가 안정적으로
분류되는지를 확인하는 실험입니다.

주의
----
이 코드의 test는 Final_Training Images 내부에서 Track 단위로 분리한
'개발용 내부 테스트 세트'입니다.

최종 10개 클래스를 선정하고 모델 설정까지 확정한 다음에는,
GTSRB 공식 Final Test Images를 마지막 평가에 사용하는 것이 좋습니다.
"""

# ============================================================
# 0. 라이브러리
# ============================================================

import gc
import json
import random
import warnings
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    GaussianNoise,
    Input,
    MaxPooling2D,
    RandomContrast,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# ============================================================
# 1. 사용자가 수정할 설정
# ============================================================

# ------------------------------------------------------------
# 1-1. GTSRB Final_Training/Images 경로
# ------------------------------------------------------------
#
# 아래는 예시입니다. 본인 컴퓨터의 실제 경로로 수정하세요.
#
# 예:
# DATASET_DIR = Path(
#     "/home/leeseokhyun/Downloads/GTSRB/Final_Training/Images"
# )
DATASET_DIR = Path('./GTSRB/Final_Training/Images')


# ------------------------------------------------------------
# 1-2. 후보 클래스
# ------------------------------------------------------------
#
# 원하는 후보 클래스 개수만큼 입력하세요. 현재 예시는 14개입니다.
#
# 아래 목록은 14개 후보 클래스의 예시입니다.
#
# 본인이 정한 후보 클래스가 따로 있다면 이 부분만 바꾸면 됩니다.
#
# 목록 순서대로 모델의 새 라벨 0부터 NUM_CLASSES-1까지 자동 배정됩니다.
SELECTED_CLASSES = [
    (13, "양보"),
    (14, "정지"),
    (17, "진입 금지"),
    (18, "일반 위험 도로"),
    (25, "도로 공사 중"),
    (28, "어린이 횡단 주의"),
    (33, "우회전 지시"),
    (34, "좌회전 지시"),
    (35, "직진 지시"),
    (40, "회전교차로"),
]


# ------------------------------------------------------------
# 1-3. 세 번의 Track 분할 seed
# ------------------------------------------------------------
#
# 클래스 안정성을 확인하기 위해 SPLIT_SEED만 바꿉니다.
SPLIT_SEEDS = [42, 100, 2026]

# CNN 초기 가중치, 데이터 셔플, 증강 조건은 고정합니다.
TRAINING_SEED = 100


# ------------------------------------------------------------
# 1-4. 이미지와 데이터 분할 설정
# ------------------------------------------------------------

IMAGE_SIZE = 28

# ------------------------------------------------------------
# 입력 이미지 색상 설정
# ------------------------------------------------------------
#
# 이번 코드는 RGB 컬러 입력을 사용합니다.
#
# RGB:
#   이미지 shape = (28, 28, 3)
#
# 흑백으로 되돌리고 싶다면:
#   INPUT_COLOR_MODE = "grayscale"
#
# 로 바꾸면 되도록 전처리와 모델 입력을 자동화했습니다.
INPUT_COLOR_MODE = "grayscale"

if INPUT_COLOR_MODE == "rgb":
    INPUT_CHANNELS = 3
elif INPUT_COLOR_MODE == "grayscale":
    INPUT_CHANNELS = 1
else:
    raise ValueError(
        "INPUT_COLOR_MODE는 'rgb' 또는 'grayscale'이어야 합니다."
    )

# 현재 사용하던 구조와 동일하게:
# 학습 70%, 검증 15%, 내부 테스트 15% 정도로 Track을 분리합니다.
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# ROI 주변 여백 비율입니다.
#
# 기존 실험과 동일한 조건으로 비교하려면 0.0을 유지하세요.
# 나중에 ROI 여백 실험을 할 때만 0.05~0.08 정도로 변경합니다.
ROI_MARGIN_RATIO = 0.0

# 너무 작은 ROI만 제외합니다.
MIN_ROI_SIZE = 8


# ------------------------------------------------------------
# 1-5. 학습 설정
# ------------------------------------------------------------

BATCH_SIZE = 64

# 최대 Epoch입니다.
# EarlyStopping 조건을 만족하면 100 이전에 자동 종료됩니다.
EPOCHS = 100

LEARNING_RATE = 0.001

USE_DATA_AUGMENTATION = True


# ------------------------------------------------------------
# 1-6. CNN 모델 구조 설정
# ------------------------------------------------------------
#
# 모델 구조를 바꾸고 싶을 때 build_model() 내부를 직접 수정하지 않고
# 아래 세 값만 변경하면 됩니다.
#
# 기존 구조:
#   CONV1_FILTERS = 5
#   CONV2_FILTERS = 6
#   DENSE_UNITS   = 16
#
# 추천 확장 구조:
#   CONV1_FILTERS = 6
#   CONV2_FILTERS = 8
#   DENSE_UNITS   = 24
#
# 조금 더 큰 비교 구조:
#   CONV1_FILTERS = 8
#   CONV2_FILTERS = 12
#   DENSE_UNITS   = 24
CONV1_FILTERS = 5
CONV2_FILTERS = 6
DENSE_UNITS = 16

# 파일명과 결과 폴더에 사용할 모델 구조 태그입니다.
# 예: rgb_c5_c6_d16
MODEL_TAG = (
    f"{INPUT_COLOR_MODE}_"
    f"c{CONV1_FILTERS}_"
    f"c{CONV2_FILTERS}_"
    f"d{DENSE_UNITS}"
)

# EarlyStopping 설정
EARLY_STOPPING_PATIENCE = 12
EARLY_STOPPING_MIN_DELTA = 0.002


# ------------------------------------------------------------
# 1-6. 결과 저장 경로
# ------------------------------------------------------------

# 실제 SELECTED_CLASSES 개수에 따라 결과 폴더명이 자동으로 정해집니다.
# 예: 14개이면 gtsrb_14class_three_seed_results
OUTPUT_DIR = Path(
    f"./gtsrb_{len(SELECTED_CLASSES)}class_"
    f"{MODEL_TAG}_three_seed_results"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 클래스 매핑 생성 및 설정 검사
# ============================================================

NUM_CLASSES = len(SELECTED_CLASSES)

if NUM_CLASSES < 2:
    raise ValueError(
        "SELECTED_CLASSES에는 최소 2개 이상의 클래스가 필요합니다. "
        f"현재 클래스 수: {NUM_CLASSES}"
    )

original_class_ids = [
    original_id
    for original_id, _ in SELECTED_CLASSES
]

if len(set(original_class_ids)) != NUM_CLASSES:
    raise ValueError(
        "SELECTED_CLASSES에 중복된 GTSRB ClassId가 있습니다."
    )

ORIGINAL_TO_NEW_LABEL = {
    original_id: new_label
    for new_label, (original_id, _) in enumerate(SELECTED_CLASSES)
}

NEW_LABEL_TO_ORIGINAL = {
    new_label: original_id
    for original_id, new_label in ORIGINAL_TO_NEW_LABEL.items()
}

NEW_LABEL_TO_NAME = {
    new_label: class_name
    for new_label, (_, class_name) in enumerate(SELECTED_CLASSES)
}


# ============================================================
# 3. 공통 보조 함수
# ============================================================

def set_training_seed(seed: int) -> None:
    """
    CNN 학습과 관련된 난수를 고정합니다.

    이 함수는 각 반복 학습을 시작하기 전에 호출됩니다.

    TRAINING_SEED가 동일하므로:
    - 초기 가중치 생성 규칙
    - 학습 데이터 셔플 규칙
    - 데이터 증강 난수 규칙

    을 가능한 한 동일하게 유지합니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    # TensorFlow 연산을 가능한 한 재현 가능하게 만듭니다.
    # 환경에 따라 지원되지 않을 수 있으므로 예외를 허용합니다.
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def find_column(dataframe: pd.DataFrame, wanted_name: str) -> str:
    """
    CSV 열 이름을 대소문자 구분 없이 찾습니다.
    """
    lower_to_original = {
        str(column).lower(): str(column)
        for column in dataframe.columns
    }

    key = wanted_name.lower()

    if key not in lower_to_original:
        raise KeyError(
            f"CSV에서 '{wanted_name}' 열을 찾지 못했습니다.\n"
            f"실제 열 목록: {list(dataframe.columns)}"
        )

    return lower_to_original[key]


def get_track_id(filename: str) -> str:
    """
    XXXXX_YYYYY.ppm 파일명에서 XXXXX 부분을 Track ID로 사용합니다.

    동일한 Track ID는 같은 실제 표지판을 연속 촬영한 이미지입니다.
    """
    return Path(filename).stem.split("_")[0]


def expand_roi_with_margin(
    image_width: int,
    image_height: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    margin_ratio: float,
):
    """
    ROI 좌표에 선택적으로 여백을 추가합니다.

    ROI_MARGIN_RATIO가 0.0이면 원래 ROI를 그대로 사용합니다.
    """
    roi_width = x2 - x1 + 1
    roi_height = y2 - y1 + 1

    margin_x = int(round(roi_width * margin_ratio))
    margin_y = int(round(roi_height * margin_ratio))

    expanded_x1 = max(0, x1 - margin_x)
    expanded_y1 = max(0, y1 - margin_y)
    expanded_x2 = min(image_width - 1, x2 + margin_x)
    expanded_y2 = min(image_height - 1, y2 + margin_y)

    return expanded_x1, expanded_y1, expanded_x2, expanded_y2


def resize_with_padding(
    image: np.ndarray,
    target_size: int = 28,
) -> np.ndarray:
    """
    가로세로 비율을 유지하면서 이미지를 target_size × target_size로 맞춥니다.

    이 함수는 다음 두 입력을 모두 처리할 수 있습니다.

    - 흑백: (높이, 너비)
    - RGB : (높이, 너비, 3)

    처리 방식
    ---------
    - 작은 이미지는 INTER_CUBIC으로 확대
    - 큰 이미지는 INTER_AREA로 축소
    - 남는 부분은 검은색 0으로 채움
    """
    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError("크기가 0인 잘못된 이미지입니다.")

    scale = min(
        target_size / width,
        target_size / height,
    )

    new_width = max(
        1,
        int(round(width * scale)),
    )
    new_height = max(
        1,
        int(round(height * scale)),
    )

    interpolation = (
        cv2.INTER_CUBIC
        if scale > 1.0
        else cv2.INTER_AREA
    )

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation,
    )

    # 흑백과 RGB에 맞는 검은색 캔버스를 생성합니다.
    if image.ndim == 2:
        canvas = np.zeros(
            (target_size, target_size),
            dtype=np.uint8,
        )
    elif image.ndim == 3:
        channel_count = image.shape[2]

        canvas = np.zeros(
            (
                target_size,
                target_size,
                channel_count,
            ),
            dtype=np.uint8,
        )
    else:
        raise ValueError(
            f"지원하지 않는 이미지 차원입니다: {image.shape}"
        )

    x_start = (
        target_size - new_width
    ) // 2
    y_start = (
        target_size - new_height
    ) // 2

    canvas[
        y_start:y_start + new_height,
        x_start:x_start + new_width,
        ...
    ] = resized

    return canvas


def preprocess_one_image(
    image_path: Path,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray | None:
    """
    이미지 한 장을 설정에 따라 28×28 RGB 또는 흑백으로 전처리합니다.
    """
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        warnings.warn(f"이미지를 읽지 못했습니다: {image_path}")
        return None

    image_height, image_width = image.shape[:2]

    # CSV 좌표를 이미지 범위 안으로 제한합니다.
    x1 = max(0, min(int(x1), image_width - 1))
    y1 = max(0, min(int(y1), image_height - 1))
    x2 = max(0, min(int(x2), image_width - 1))
    y2 = max(0, min(int(y2), image_height - 1))

    if x2 < x1 or y2 < y1:
        warnings.warn(f"잘못된 ROI 좌표: {image_path}")
        return None

    # 필요하면 ROI에 여백을 추가합니다.
    x1, y1, x2, y2 = expand_roi_with_margin(
        image_width=image_width,
        image_height=image_height,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        margin_ratio=ROI_MARGIN_RATIO,
    )

    roi = image[y1:y2 + 1, x1:x2 + 1]

    if roi.size == 0:
        warnings.warn(f"ROI가 비어 있습니다: {image_path}")
        return None

    roi_height, roi_width = roi.shape[:2]

    if roi_width < MIN_ROI_SIZE or roi_height < MIN_ROI_SIZE:
        warnings.warn(
            f"ROI가 너무 작아 제외합니다: {image_path} "
            f"({roi_width}×{roi_height})"
        )
        return None

    # OpenCV는 컬러 이미지를 BGR 순서로 읽습니다.
    # TensorFlow/Keras에서 일반적으로 사용하는 RGB 순서로 변환합니다.
    if INPUT_COLOR_MODE == "rgb":
        processed_image = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2RGB,
        )

    else:
        processed_image = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY,
        )

    return resize_with_padding(
        image=processed_image,
        target_size=IMAGE_SIZE,
    )


# ============================================================
# 4. 선택한 후보 클래스 전체를 한 번만 전처리
# ============================================================

def load_all_selected_images():
    """
    선택한 후보 클래스의 이미지를 모두 읽고 전처리합니다.

    세 번의 실험에서 이미지 전처리 결과는 동일하므로,
    매 실험마다 PPM 파일을 다시 읽지 않고 처음 한 번만 처리합니다.

    반환값
    ------
    images:
        RGB이면 (전체 이미지 수, 28, 28, 3), 흑백이면 (전체 이미지 수, 28, 28, 1)

    labels:
        새 라벨 0~14

    tracks:
        Track ID

    original_ids:
        GTSRB 원본 ClassId

    filenames:
        원본 파일명
    """
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            "GTSRB Images 폴더를 찾지 못했습니다.\n"
            f"현재 DATASET_DIR: {DATASET_DIR}\n"
            "코드 상단의 DATASET_DIR을 실제 경로로 수정하세요."
        )

    images = []
    labels = []
    tracks = []
    original_ids = []
    filenames = []

    print("=" * 90)
    print("후보 클래스 이미지 전처리")
    print("=" * 90)

    for new_label, (original_class_id, class_name) in enumerate(
        SELECTED_CLASSES
    ):
        class_dir = DATASET_DIR / f"{original_class_id:05d}"
        csv_path = class_dir / f"GT-{original_class_id:05d}.csv"

        if not class_dir.exists():
            raise FileNotFoundError(
                f"클래스 폴더가 없습니다: {class_dir}"
            )

        if not csv_path.exists():
            raise FileNotFoundError(
                f"클래스 CSV가 없습니다: {csv_path}"
            )

        annotations = pd.read_csv(
            csv_path,
            sep=";",
        )

        filename_column = find_column(
            annotations,
            "Filename",
        )
        x1_column = find_column(
            annotations,
            "Roi.X1",
        )
        y1_column = find_column(
            annotations,
            "Roi.Y1",
        )
        x2_column = find_column(
            annotations,
            "Roi.X2",
        )
        y2_column = find_column(
            annotations,
            "Roi.Y2",
        )

        used_count = 0
        skipped_count = 0
        class_track_ids = set()

        for _, row in annotations.iterrows():
            filename = str(row[filename_column])
            image_path = class_dir / filename
            track_id = get_track_id(filename)

            processed = preprocess_one_image(
                image_path=image_path,
                x1=row[x1_column],
                y1=row[y1_column],
                x2=row[x2_column],
                y2=row[y2_column],
            )

            if processed is None:
                skipped_count += 1
                continue

            images.append(processed)
            labels.append(new_label)
            tracks.append(track_id)
            original_ids.append(original_class_id)
            filenames.append(filename)

            class_track_ids.add(track_id)
            used_count += 1

        print(
            f"새 라벨 {new_label:2d} | "
            f"GTSRB {original_class_id:2d} | "
            f"{class_name:<14} | "
            f"이미지 {used_count:4d}장 | "
            f"Track {len(class_track_ids):2d}개 | "
            f"제외 {skipped_count:3d}장"
        )

    images = np.asarray(
        images,
        dtype=np.float32,
    )

    # 흑백 이미지만 채널 차원을 추가합니다.
    #
    # 흑백:
    #   (N, 28, 28) -> (N, 28, 28, 1)
    #
    # RGB:
    #   이미 (N, 28, 28, 3)이므로 그대로 사용
    if INPUT_COLOR_MODE == "grayscale":
        images = np.expand_dims(
            images,
            axis=-1,
        )

    # 0~255 정수 픽셀을 0~1 실수 범위로 정규화합니다.
    images = images / 255.0

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )
    tracks = np.asarray(
        tracks,
        dtype=str,
    )
    original_ids = np.asarray(
        original_ids,
        dtype=np.int64,
    )
    filenames = np.asarray(
        filenames,
        dtype=str,
    )

    print("\n전처리 완료")
    print(f"images.shape: {images.shape}")
    print(f"labels.shape: {labels.shape}")
    print(
        f"픽셀 범위: {images.min():.3f} ~ {images.max():.3f}"
    )

    return images, labels, tracks, original_ids, filenames


# ============================================================
# 5. SPLIT_SEED에 따라 Track 단위 분할
# ============================================================

def split_one_class_tracks(
    unique_tracks,
    split_seed: int,
    original_class_id: int,
):
    """
    한 클래스의 Track들을 학습/검증/내부 테스트로 분리합니다.

    클래스마다 같은 셔플 순서가 반복되지 않도록
    split_seed + original_class_id를 사용합니다.
    """
    unique_tracks = sorted(set(unique_tracks))

    if len(unique_tracks) < 3:
        raise ValueError(
            f"GTSRB ClassId {original_class_id}의 Track이 "
            f"{len(unique_tracks)}개뿐이어서 3개 세트로 나눌 수 없습니다."
        )

    rng = random.Random(
        split_seed + original_class_id
    )
    rng.shuffle(unique_tracks)

    track_count = len(unique_tracks)

    validation_count = max(
        1,
        round(track_count * VALIDATION_RATIO),
    )
    test_count = max(
        1,
        round(track_count * TEST_RATIO),
    )
    train_count = (
        track_count
        - validation_count
        - test_count
    )

    # 학습 Track이 최소 1개는 남도록 조정합니다.
    while train_count < 1:
        if validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise ValueError(
                f"GTSRB ClassId {original_class_id}를 "
                "학습/검증/테스트로 나눌 수 없습니다."
            )

        train_count = (
            track_count
            - validation_count
            - test_count
        )

    train_tracks = set(
        unique_tracks[:train_count]
    )

    validation_start = train_count
    validation_end = (
        train_count + validation_count
    )

    validation_tracks = set(
        unique_tracks[
            validation_start:validation_end
        ]
    )

    test_tracks = set(
        unique_tracks[validation_end:]
    )

    return (
        train_tracks,
        validation_tracks,
        test_tracks,
    )


def make_split_indices(
    labels: np.ndarray,
    tracks: np.ndarray,
    split_seed: int,
):
    """
    선택한 모든 클래스에 대해 Track 단위 분할 인덱스를 만듭니다.
    """
    train_indices = []
    validation_indices = []
    test_indices = []

    split_rows = []

    for new_label, (original_class_id, class_name) in enumerate(
        SELECTED_CLASSES
    ):
        class_indices = np.where(
            labels == new_label
        )[0]

        class_tracks = tracks[class_indices]
        unique_tracks = np.unique(class_tracks)

        (
            train_tracks,
            validation_tracks,
            test_tracks,
        ) = split_one_class_tracks(
            unique_tracks=unique_tracks,
            split_seed=split_seed,
            original_class_id=original_class_id,
        )

        class_train_indices = [
            index
            for index in class_indices
            if tracks[index] in train_tracks
        ]

        class_validation_indices = [
            index
            for index in class_indices
            if tracks[index] in validation_tracks
        ]

        class_test_indices = [
            index
            for index in class_indices
            if tracks[index] in test_tracks
        ]

        train_indices.extend(
            class_train_indices
        )
        validation_indices.extend(
            class_validation_indices
        )
        test_indices.extend(
            class_test_indices
        )

        split_rows.append(
            {
                "new_label": new_label,
                "gtsrb_class_id": original_class_id,
                "class_name": class_name,
                "total_tracks": len(unique_tracks),
                "train_tracks": len(train_tracks),
                "validation_tracks": len(validation_tracks),
                "test_tracks": len(test_tracks),
                "train_images": len(class_train_indices),
                "validation_images": len(class_validation_indices),
                "test_images": len(class_test_indices),
            }
        )

    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(validation_indices, dtype=np.int64),
        np.asarray(test_indices, dtype=np.int64),
        pd.DataFrame(split_rows),
    )


# ============================================================
# 6. 데이터 증강과 tf.data
# ============================================================

def create_data_augmentation(
    training_seed: int,
):
    """
    세 번의 실험에서 동일한 증강 설정을 사용합니다.

    좌우 반전은 좌회전/우회전 의미를 바꾸므로 사용하지 않습니다.
    """
    return Sequential(
        [
            RandomRotation(
                factor=0.03,
                fill_mode="constant",
                fill_value=0.0,
                seed=training_seed + 1,
            ),
            RandomTranslation(
                height_factor=0.08,
                width_factor=0.08,
                fill_mode="constant",
                fill_value=0.0,
                seed=training_seed + 2,
            ),
            RandomZoom(
                height_factor=(-0.10, 0.10),
                width_factor=(-0.10, 0.10),
                fill_mode="constant",
                fill_value=0.0,
                seed=training_seed + 3,
            ),
            RandomContrast(
                factor=0.15,
                seed=training_seed + 4,
            ),
            GaussianNoise(
                stddev=0.02,
                seed=training_seed + 5,
            ),
        ],
        name="training_data_augmentation",
    )


def make_tf_datasets(
    x_train,
    y_train,
    x_validation,
    y_validation,
    class_weight_dictionary,
    training_seed: int,
):
    """
    학습 및 검증 TensorFlow Dataset을 생성합니다.
    """
    sample_weights = np.asarray(
        [
            class_weight_dictionary[int(label)]
            for label in y_train
        ],
        dtype=np.float32,
    )

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (
            x_train,
            y_train,
            sample_weights,
        )
    )

    train_dataset = train_dataset.shuffle(
        buffer_size=len(x_train),
        seed=training_seed,
        reshuffle_each_iteration=True,
    )

    train_dataset = train_dataset.batch(
        BATCH_SIZE
    )

    if USE_DATA_AUGMENTATION:
        augmentation = create_data_augmentation(
            training_seed=training_seed,
        )

        def augment_batch(
            batch_images,
            batch_labels,
            batch_weights,
        ):
            batch_images = augmentation(
                batch_images,
                training=True,
            )

            batch_images = tf.clip_by_value(
                batch_images,
                0.0,
                1.0,
            )

            return (
                batch_images,
                batch_labels,
                batch_weights,
            )

        train_dataset = train_dataset.map(
            augment_batch,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    train_dataset = train_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    validation_dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                x_validation,
                y_validation,
            )
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_dataset, validation_dataset


# ============================================================
# 7. 가변 출력 소형 CNN
# ============================================================

def build_model(
    num_classes: int,
) -> tf.keras.Model:
    """
    기존 Basys3용 소형 CNN 구조를 유지하고
    마지막 출력만 후보 15개에 맞춥니다.

    크기 흐름:
    28×28×INPUT_CHANNELS
    -> Conv CONV1_FILTERS
    -> Pool
    -> Conv CONV2_FILTERS
    -> Pool
    -> 7×7×CONV2_FILTERS
    -> Dense DENSE_UNITS
    -> Dense NUM_CLASSES
    """
    model = Sequential(
        [
            Input(
                shape=(
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    INPUT_CHANNELS,
                ),
                name="input_image",
            ),

            Conv2D(
                filters=CONV1_FILTERS,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                name="conv2d",
            ),

            MaxPooling2D(
                pool_size=(2, 2),
                padding="same",
                name="max_pooling2d",
            ),

            Conv2D(
                filters=CONV2_FILTERS,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                name="conv2d_1",
            ),

            MaxPooling2D(
                pool_size=(2, 2),
                padding="same",
                name="max_pooling2d_1",
            ),

            Flatten(
                name="flatten",
            ),

            Dense(
                units=DENSE_UNITS,
                activation="relu",
                name="dense",
            ),

            Dense(
                units=num_classes,
                activation="softmax",
                name="dense_1",
            ),
        ],
        name=(
            f"gtsrb_{num_classes}class_"
            f"{MODEL_TAG}_micro_cnn"
        ),
    )

    model.compile(
        optimizer=Adam(
            learning_rate=LEARNING_RATE,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# 8. 실행별 결과 저장 함수
# ============================================================

def save_history_plots(
    history,
    run_dir: Path,
    run_tag: str,
):
    """
    학습/검증 Accuracy와 Loss 그래프를 저장합니다.
    """
    history_data = history.history

    plt.figure(figsize=(8, 5))
    plt.plot(
        history_data["accuracy"],
        label="train_accuracy",
    )
    plt.plot(
        history_data["val_accuracy"],
        label="validation_accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        f"Training and Validation Accuracy\n{run_tag}"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    accuracy_path = (
        run_dir
        / f"training_accuracy_{run_tag}.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        history_data["loss"],
        label="train_loss",
    )
    plt.plot(
        history_data["val_loss"],
        label="validation_loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"Training and Validation Loss\n{run_tag}"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_path = (
        run_dir
        / f"training_loss_{run_tag}.png"
    )

    plt.savefig(
        loss_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def save_confusion_outputs(
    y_true,
    y_pred,
    run_dir: Path,
    run_tag: str,
):
    """
    혼동행렬을 다음 네 가지 형태로 저장합니다.

    1. 원본 개수 PNG
    2. 원본 개수 CSV
    3. 행 기준 비율 PNG
    4. 행 기준 비율 CSV

    비율 혼동행렬은 클래스별 테스트 이미지 수가 다를 때
    비교하기 더 편합니다.
    """
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    row_sums = matrix.sum(
        axis=1,
        keepdims=True,
    )

    normalized_matrix = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(
            matrix,
            dtype=np.float64,
        ),
        where=row_sums != 0,
    )

    row_names = [
        (
            f"true_{label}_"
            f"gtsrb_{NEW_LABEL_TO_ORIGINAL[label]}_"
            f"{NEW_LABEL_TO_NAME[label]}"
        )
        for label in range(NUM_CLASSES)
    ]

    column_names = [
        (
            f"pred_{label}_"
            f"gtsrb_{NEW_LABEL_TO_ORIGINAL[label]}_"
            f"{NEW_LABEL_TO_NAME[label]}"
        )
        for label in range(NUM_CLASSES)
    ]

    raw_dataframe = pd.DataFrame(
        matrix,
        index=row_names,
        columns=column_names,
    )

    normalized_dataframe = pd.DataFrame(
        normalized_matrix,
        index=row_names,
        columns=column_names,
    )

    raw_csv_path = (
        run_dir
        / f"confusion_matrix_raw_{run_tag}.csv"
    )

    normalized_csv_path = (
        run_dir
        / f"confusion_matrix_normalized_{run_tag}.csv"
    )

    raw_dataframe.to_csv(
        raw_csv_path,
        encoding="utf-8-sig",
    )

    normalized_dataframe.to_csv(
        normalized_csv_path,
        encoding="utf-8-sig",
        float_format="%.4f",
    )

    # 원본 개수 혼동행렬 그림
    plt.figure(figsize=(13, 11))
    plt.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
    )
    plt.title(
        f"Confusion Matrix - Raw Counts\n{run_tag}"
    )
    plt.colorbar()

    tick_labels = [
        f"{label}\nID{NEW_LABEL_TO_ORIGINAL[label]}"
        for label in range(NUM_CLASSES)
    ]

    plt.xticks(
        np.arange(NUM_CLASSES),
        tick_labels,
        rotation=45,
        ha="right",
    )
    plt.yticks(
        np.arange(NUM_CLASSES),
        tick_labels,
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    threshold = (
        matrix.max() / 2.0
        if matrix.max() > 0
        else 0
    )

    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            plt.text(
                column,
                row,
                str(matrix[row, column]),
                horizontalalignment="center",
                fontsize=7,
                color=(
                    "white"
                    if matrix[row, column] > threshold
                    else "black"
                ),
            )

    plt.tight_layout()

    raw_png_path = (
        run_dir
        / f"confusion_matrix_raw_{run_tag}.png"
    )

    plt.savefig(
        raw_png_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    # 행 기준 비율 혼동행렬 그림
    plt.figure(figsize=(13, 11))
    plt.imshow(
        normalized_matrix,
        interpolation="nearest",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    plt.title(
        f"Confusion Matrix - Row Normalized\n{run_tag}"
    )
    plt.colorbar()

    plt.xticks(
        np.arange(NUM_CLASSES),
        tick_labels,
        rotation=45,
        ha="right",
    )
    plt.yticks(
        np.arange(NUM_CLASSES),
        tick_labels,
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            value = normalized_matrix[row, column]

            plt.text(
                column,
                row,
                f"{value:.2f}",
                horizontalalignment="center",
                fontsize=7,
                color=(
                    "white"
                    if value > 0.5
                    else "black"
                ),
            )

    plt.tight_layout()

    normalized_png_path = (
        run_dir
        / f"confusion_matrix_normalized_{run_tag}.png"
    )

    plt.savefig(
        normalized_png_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    return {
        "raw_csv": raw_csv_path,
        "normalized_csv": normalized_csv_path,
        "raw_png": raw_png_path,
        "normalized_png": normalized_png_path,
    }


def save_classification_report(
    y_true,
    y_pred,
    test_loss: float,
    test_accuracy: float,
    split_seed: int,
    training_seed: int,
    run_dir: Path,
    run_tag: str,
):
    """
    분류 보고서를 TXT와 CSV로 저장합니다.

    CSV는 나중에 세 번의 F1-score를 자동 집계하는 데 사용합니다.
    """
    target_names = [
        (
            f"{label}: GTSRB "
            f"{NEW_LABEL_TO_ORIGINAL[label]} "
            f"{NEW_LABEL_TO_NAME[label]}"
        )
        for label in range(NUM_CLASSES)
    ]

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    txt_path = (
        run_dir
        / f"classification_report_{run_tag}.txt"
    )

    with txt_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            f"GTSRB {NUM_CLASSES}-class 28x28 {INPUT_COLOR_MODE} CNN\n"
        )
        file.write("=" * 70 + "\n\n")
        file.write(f"SPLIT_SEED: {split_seed}\n")
        file.write(f"TRAINING_SEED: {training_seed}\n")
        file.write(f"RUN_TAG: {run_tag}\n\n")
        file.write(f"Test loss: {test_loss:.6f}\n")
        file.write(
            f"Test accuracy: {test_accuracy:.6f}\n\n"
        )
        file.write(report_text)

    class_rows = []

    for label in range(NUM_CLASSES):
        target_name = target_names[label]
        metrics = report_dict[target_name]

        class_rows.append(
            {
                "new_label": label,
                "gtsrb_class_id": NEW_LABEL_TO_ORIGINAL[label],
                "class_name": NEW_LABEL_TO_NAME[label],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1-score"],
                "support": int(metrics["support"]),
                "split_seed": split_seed,
                "training_seed": training_seed,
                "run_tag": run_tag,
            }
        )

    class_metrics_dataframe = pd.DataFrame(
        class_rows
    )

    csv_path = (
        run_dir
        / f"classification_metrics_{run_tag}.csv"
    )

    class_metrics_dataframe.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    return (
        report_text,
        report_dict,
        class_metrics_dataframe,
        txt_path,
        csv_path,
    )


# ============================================================
# 9. 한 개 SPLIT_SEED 학습
# ============================================================

def run_one_experiment(
    images: np.ndarray,
    labels: np.ndarray,
    tracks: np.ndarray,
    split_seed: int,
    training_seed: int,
):
    """
    지정된 SPLIT_SEED로 한 번 학습하고 모든 결과를 저장합니다.
    """
    # 이전 학습 모델과 TensorFlow 그래프를 정리합니다.
    tf.keras.backend.clear_session()
    gc.collect()

    set_training_seed(
        training_seed
    )

    run_tag = (
        f"{MODEL_TAG}_"
        f"split{split_seed}_"
        f"train{training_seed}"
    )

    run_dir = OUTPUT_DIR / run_tag
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "#" * 90)
    print(f"실험 시작: {run_tag}")
    print("#" * 90)

    (
        train_indices,
        validation_indices,
        test_indices,
        split_dataframe,
    ) = make_split_indices(
        labels=labels,
        tracks=tracks,
        split_seed=split_seed,
    )

    # 재현성을 위해 인덱스를 정렬합니다.
    train_indices = np.sort(train_indices)
    validation_indices = np.sort(validation_indices)
    test_indices = np.sort(test_indices)

    x_train = images[train_indices]
    y_train = labels[train_indices]

    x_validation = images[validation_indices]
    y_validation = labels[validation_indices]

    x_test = images[test_indices]
    y_test = labels[test_indices]

    print(f"x_train      : {x_train.shape}")
    print(f"x_validation : {x_validation.shape}")
    print(f"x_test       : {x_test.shape}")

    split_csv_path = (
        run_dir
        / f"track_split_summary_{run_tag}.csv"
    )

    split_dataframe.to_csv(
        split_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[저장] Track 분할 정보: {split_csv_path}")

    # 클래스별 이미지 수 출력
    for split_name, split_labels in [
        ("train", y_train),
        ("validation", y_validation),
        ("test", y_test),
    ]:
        count_dictionary = Counter(
            split_labels.tolist()
        )

        print(f"\n[{split_name} 클래스별 이미지 수]")

        for label in range(NUM_CLASSES):
            print(
                f"라벨 {label:2d} | "
                f"GTSRB {NEW_LABEL_TO_ORIGINAL[label]:2d} | "
                f"{count_dictionary.get(label, 0):4d}장"
            )

    # 학습 클래스 가중치
    class_ids = np.arange(NUM_CLASSES)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_ids,
        y=y_train,
    )

    class_weight_dictionary = {
        int(class_id): float(weight)
        for class_id, weight in zip(
            class_ids,
            weights,
        )
    }

    train_dataset, validation_dataset = make_tf_datasets(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        class_weight_dictionary=class_weight_dictionary,
        training_seed=training_seed,
    )

    model = build_model(
        num_classes=NUM_CLASSES,
    )

    print("\n[모델 구조]")
    model.summary()

    best_model_path = (
        run_dir
        / f"best_model_{run_tag}.keras"
    )

    callbacks = [
        ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),

        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA,
            restore_best_weights=True,
            verbose=1,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=4,
            min_delta=0.001,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    probabilities = model.predict(
        x_test,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    y_pred = np.argmax(
        probabilities,
        axis=1,
    )

    (
        report_text,
        report_dict,
        class_metrics_dataframe,
        report_txt_path,
        report_csv_path,
    ) = save_classification_report(
        y_true=y_test,
        y_pred=y_pred,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        split_seed=split_seed,
        training_seed=training_seed,
        run_dir=run_dir,
        run_tag=run_tag,
    )

    print("\n[클래스별 분류 보고서]")
    print(report_text)

    print(f"[저장] 보고서 TXT: {report_txt_path}")
    print(f"[저장] 지표 CSV  : {report_csv_path}")

    confusion_paths = save_confusion_outputs(
        y_true=y_test,
        y_pred=y_pred,
        run_dir=run_dir,
        run_tag=run_tag,
    )

    for name, path in confusion_paths.items():
        print(f"[저장] {name}: {path}")

    save_history_plots(
        history=history,
        run_dir=run_dir,
        run_tag=run_tag,
    )

    final_model_path = (
        run_dir
        / (
            f"cnn_gtsrb_{NUM_CLASSES}class_{run_tag}_"
            f"acc_{test_accuracy:.4f}.keras"
        )
    )

    model.save(
        final_model_path
    )

    config_path = (
        run_dir
        / f"experiment_config_{run_tag}.json"
    )

    config_data = {
        "split_seed": split_seed,
        "training_seed": training_seed,
        "num_classes": NUM_CLASSES,
        "image_size": IMAGE_SIZE,
        "input_color_mode": INPUT_COLOR_MODE,
        "input_channels": INPUT_CHANNELS,
        "validation_ratio": VALIDATION_RATIO,
        "test_ratio": TEST_RATIO,
        "roi_margin_ratio": ROI_MARGIN_RATIO,
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "epochs_ran": len(history.history["loss"]),
        "learning_rate": LEARNING_RATE,
        "conv1_filters": CONV1_FILTERS,
        "conv2_filters": CONV2_FILTERS,
        "dense_units": DENSE_UNITS,
        "model_tag": MODEL_TAG,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "best_validation_accuracy": float(
            max(history.history["val_accuracy"])
        ),
        "selected_classes": [
            {
                "new_label": label,
                "gtsrb_class_id": original_id,
                "class_name": class_name,
            }
            for label, (original_id, class_name)
            in enumerate(SELECTED_CLASSES)
        ],
    }

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    overall_row = {
        "run_tag": run_tag,
        "model_tag": MODEL_TAG,
        "conv1_filters": CONV1_FILTERS,
        "conv2_filters": CONV2_FILTERS,
        "dense_units": DENSE_UNITS,
        "input_color_mode": INPUT_COLOR_MODE,
        "input_channels": INPUT_CHANNELS,
        "split_seed": split_seed,
        "training_seed": training_seed,
        "epochs_ran": len(history.history["loss"]),
        "best_validation_accuracy": max(
            history.history["val_accuracy"]
        ),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "macro_precision": report_dict["macro avg"]["precision"],
        "macro_recall": report_dict["macro avg"]["recall"],
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"],
    }

    print("\n[실험 완료]")
    print(f"RUN_TAG: {run_tag}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(
        f"Macro F1: "
        f"{report_dict['macro avg']['f1-score']:.4f}"
    )

    return (
        class_metrics_dataframe,
        overall_row,
    )


# ============================================================
# 10. 세 번의 결과 통합
# ============================================================

def make_combined_class_summary(
    all_class_metrics,
):
    """
    클래스별로 세 번의 Precision/Recall/F1을 합치고
    평균, 최저, 최고, 변동 폭을 계산합니다.
    """
    base_dataframe = pd.DataFrame(
        [
            {
                "new_label": label,
                "gtsrb_class_id": original_id,
                "class_name": class_name,
            }
            for label, (original_id, class_name)
            in enumerate(SELECTED_CLASSES)
        ]
    )

    result = base_dataframe.copy()

    precision_columns = []
    recall_columns = []
    f1_columns = []

    for metrics_dataframe in all_class_metrics:
        split_seed = int(
            metrics_dataframe["split_seed"].iloc[0]
        )
        training_seed = int(
            metrics_dataframe["training_seed"].iloc[0]
        )

        seed_tag = (
            f"split{split_seed}_"
            f"train{training_seed}"
        )

        selected = metrics_dataframe[
            [
                "new_label",
                "precision",
                "recall",
                "f1_score",
            ]
        ].copy()

        precision_column = (
            f"precision_{seed_tag}"
        )
        recall_column = (
            f"recall_{seed_tag}"
        )
        f1_column = (
            f"f1_{seed_tag}"
        )

        selected = selected.rename(
            columns={
                "precision": precision_column,
                "recall": recall_column,
                "f1_score": f1_column,
            }
        )

        result = result.merge(
            selected,
            on="new_label",
            how="left",
        )

        precision_columns.append(
            precision_column
        )
        recall_columns.append(
            recall_column
        )
        f1_columns.append(
            f1_column
        )

    result["precision_mean"] = (
        result[precision_columns].mean(axis=1)
    )
    result["recall_mean"] = (
        result[recall_columns].mean(axis=1)
    )

    result["f1_mean"] = (
        result[f1_columns].mean(axis=1)
    )
    result["f1_min"] = (
        result[f1_columns].min(axis=1)
    )
    result["f1_max"] = (
        result[f1_columns].max(axis=1)
    )
    result["f1_range"] = (
        result["f1_max"]
        - result["f1_min"]
    )

    # 간단한 후보 판단용 상태를 자동으로 표시합니다.
    #
    # 절대적인 표준은 아니며,
    # 최종 결정 때 혼동행렬과 Track 수를 함께 확인해야 합니다.
    def classify_stability(row):
        if (
            row["f1_mean"] >= 0.80
            and row["f1_min"] >= 0.70
            and row["f1_range"] <= 0.15
        ):
            return "안정적 후보"

        if (
            row["f1_mean"] < 0.70
            or row["f1_min"] < 0.60
            or row["f1_range"] > 0.20
        ):
            return "교체 검토"

        return "관찰 필요"

    result["stability_status"] = result.apply(
        classify_stability,
        axis=1,
    )

    result = result.sort_values(
        by=[
            "f1_mean",
            "f1_min",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)

    return result


# ============================================================
# 11. 메인 실행
# ============================================================

def main():
    print("=" * 90)
    print("GTSRB 후보 클래스 3회 비교 학습")
    print("=" * 90)

    print(f"SPLIT_SEEDS  : {SPLIT_SEEDS}")
    print(f"TRAINING_SEED: {TRAINING_SEED}")
    print(f"NUM_CLASSES  : {NUM_CLASSES}")
    print(f"MAX EPOCHS   : {EPOCHS}")
    print(f"MODEL_TAG    : {MODEL_TAG}")
    print(f"COLOR_MODE   : {INPUT_COLOR_MODE}")
    print(f"INPUT_CHANNELS: {INPUT_CHANNELS}")
    print(f"CONV1_FILTERS: {CONV1_FILTERS}")
    print(f"CONV2_FILTERS: {CONV2_FILTERS}")
    print(f"DENSE_UNITS  : {DENSE_UNITS}")

    print("\n[새 라벨 매핑]")

    for label, (original_id, class_name) in enumerate(
        SELECTED_CLASSES
    ):
        print(
            f"새 라벨 {label:2d} <- "
            f"GTSRB {original_id:2d}: "
            f"{class_name}"
        )

    # 선택한 모든 이미지는 처음 한 번만 읽습니다.
    (
        images,
        labels,
        tracks,
        original_ids,
        filenames,
    ) = load_all_selected_images()

    # 현재는 직접 사용하지 않지만 추후 오분류 이미지 확인에 활용할 수 있습니다.
    _ = (
        original_ids,
        filenames,
    )

    all_class_metrics = []
    overall_rows = []

    # SPLIT_SEED만 바꾸어 세 번 자동 학습합니다.
    for split_seed in SPLIT_SEEDS:
        (
            class_metrics_dataframe,
            overall_row,
        ) = run_one_experiment(
            images=images,
            labels=labels,
            tracks=tracks,
            split_seed=split_seed,
            training_seed=TRAINING_SEED,
        )

        all_class_metrics.append(
            class_metrics_dataframe
        )
        overall_rows.append(
            overall_row
        )

    # 전체 실행 요약
    overall_dataframe = pd.DataFrame(
        overall_rows
    )

    overall_path = (
        OUTPUT_DIR
        / (
            f"overall_run_summary_"
            f"{MODEL_TAG}_"
            f"train{TRAINING_SEED}.csv"
        )
    )

    overall_dataframe.to_csv(
        overall_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    # 클래스별 세 번의 결과 통합
    combined_class_summary = (
        make_combined_class_summary(
            all_class_metrics=all_class_metrics,
        )
    )

    class_summary_path = (
        OUTPUT_DIR
        / (
            f"class_stability_summary_"
            f"{MODEL_TAG}_"
            f"train{TRAINING_SEED}.csv"
        )
    )

    combined_class_summary.to_csv(
        class_summary_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    print("\n" + "=" * 90)
    print("세 번의 실험 완료")
    print("=" * 90)

    print(f"[저장] 실행별 전체 결과: {overall_path}")
    print(f"[저장] 클래스 안정성 결과: {class_summary_path}")

    print("\n[클래스 안정성 요약]")
    print(
        combined_class_summary[
            [
                "new_label",
                "gtsrb_class_id",
                "class_name",
                "f1_mean",
                "f1_min",
                "f1_max",
                "f1_range",
                "stability_status",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n가장 먼저 확인할 파일:\n"
        f"1. {class_summary_path}\n"
        "   - 클래스별 F1 평균, 최저값, 변동 폭 확인\n\n"
        f"2. {overall_path}\n"
        "   - 실행별 Test Accuracy와 Macro F1 확인\n\n"
        "3. 각 split 폴더의 confusion_matrix_normalized_*.png\n"
        "   - 낮은 F1 클래스가 어떤 클래스로 오분류되는지 확인\n"
    )


if __name__ == "__main__":
    main()
