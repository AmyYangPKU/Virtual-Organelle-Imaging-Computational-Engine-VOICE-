import napari
import numpy as np
import tifffile as tiff
import csv
from pathlib import Path
from magicgui import magicgui
from napari.layers import Image
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QComboBox,
    QVBoxLayout,
    QWidget,
)

from .generate_ccp import generate_ccp
from .generate_er import generate_er
from .generate_mito import generate_mito, generate_mito_volume
from .generate_mt import generate_mt
from .generate_npc import generate_npc
from .resolution_tools import (
    auto_psf_3d,
    convolve_same,
    convolve_volume_center_slice,
    downsample_image,
    fit_higher_lambda_directional,
    fit_sigma_xy_3d_center_slice,
    generate_psf,
    psf_cropper,
    resolution_estimation,
    save_fit_info_csv,
    save_float_tiff,
    save_parameter_fit_info_csv,
)


APP_NAME = "VOICE_app"
viewer = None
LAST_LR_SR_CONFIG = None
REFERENCE_PATH_EDIT = None
ORGANELLE_COMBO = None
DEFAULT_OUTPUT_DIR = Path.home() / "VOICE_app_outputs"


# ============================================================
# 内置参数：不会显示在 napari 界面中
# ============================================================
IMG_SIZE = 1024
BASE_SEED = 1
DIGITS = 4
DEFAULT_LAMBDA_MIN = 150.0
DEFAULT_LAMBDA_MAX = 700.0
DEFAULT_REFRACTIVE_INDEX = 1.515
DEFAULT_MAX_ITER = 20
DEFAULT_TOL_RES = 5.0
REFERENCE_SAMPLE_COUNT = 30
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
ORGANELLE_TYPES = ("MT", "CCP", "NPCs", "Actin", "OMM", "ER")
MITO_ORGANELLE_TYPES = {"OMM"}
LR_NOISE_BANK = (0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3)
POISSON_COUNT_LEVEL = 65535.0
OMM_SIGMA_XY_RANGE = (0.5, 2.0)
OMM_SIGMA_Z_MIN = 1.2
OMM_SIGMA_Z_SPAN = 0.8
OMM_PSF_SHIFT = 1
OMM_PSF_SCATTER_GAIN = 1.0
OMM_PSF_XY_SIZE = 257

# 每张图随机采样的参数范围
# 格式：(min, max)
PARAM_RANGES = {
    "numMicrotubules": (100, 150),
    "density_factor": (2.0, 5.0),
    "centerfactor": (0.4, 0.9),
    "maxLength_nm": (5000.0, 12000.0),
    "bendStrengthBase": (0.06, 0.08),
    "bendStrengthEdge": (0.1, 0.3),
    "noiseLevel": (0.3, 0.6),
    "numClusters": (3, 7),
    "controlmin": (3, 4),
    "controlmax": (5, 7),
    "addORoverlap": (0.5, 1.0),
}
CCP_DEFAULT_PARAMS = {
    "diam_min_nm": 100.0,
    "diam_max_nm": 200.0,
}
NPC_PARAM_RANGES = {
    "numNPC": (1000, 3000),
    "pair_ratio": (0.1, 0.3),
    "pairDist_nm_min": (100.0, 200.0),
    "pairDist_nm_max": (200.0, 300.0),
}
NPC_DEFAULT_PARAMS = {
    "Nup_dia_nm": 0.0,
    "minDist_nm": 50.0,
    "R_nm_min": 50.0,
    "R_nm_max": 60.0,
    "ring_thick_nm": 0.0,
    "nSym": 8,
    "ptsPerSubunit": 1,
    "dtheta_deg": 0.0,
    "sigma_r_nm": 2.0,
    "npc_gain_min": 0.3,
    "npc_gain_max": 1.2,
    "dot_sigma_nm": 0.0,
}
ER_PARAM_RANGES = {
    "diam_min_nm": (60.0, 60.0),
    "diam_max_nm": (300.0, 300.0),
    "density": (0.05, 0.08),
    "noise_level": (1.5, 1.5),
    "holes_ratio": (0.10, 0.10),
}


def sample_mito_parameters(rng):
    rmin_nm = 200 / 2
    rmax_nm = 400 / 2
    r_circle_nm = rmax_nm * (1 + 0.5 * rng.random())
    zsize = 100

    num_p = rng.random()
    if num_p > 0.7:
        num_total = int(np.floor(((400 + rng.random() * 150) / 2) + 0.5))
        long_ratio = 0.8 + rng.random() * 0.2
        num_mito = int(np.floor(num_total * long_ratio + 0.5))
        num_circle = num_total - num_mito
    else:
        num_total = int(np.floor(((250 + rng.random() * 100) / 2) + 0.5))
        long_ratio = 0.8 + rng.random() * 0.2
        num_mito = int(np.floor(num_total * long_ratio + 0.5))
        num_circle = num_total - num_mito

    same_z_p = 0.8
    len_p = rng.random()
    if len_p > 0.5:
        len_mito_min_nm = 1500.0
        len_mito_max_nm = 6000.0
    elif len_p > 0.9:
        len_mito_min_nm = 4000.0
        len_mito_max_nm = 8000.0
    else:
        len_mito_min_nm = 900.0
        len_mito_max_nm = 3000.0

    return {
        "mito_zsize": zsize,
        "mito_num_p": float(num_p),
        "mito_num_total": num_total,
        "mito_long_ratio": float(long_ratio),
        "mito_num_mito": num_mito,
        "mito_num_circle": num_circle,
        "mito_same_z_p": same_z_p,
        "mito_len_p": float(len_p),
        "mito_len_min_nm": len_mito_min_nm,
        "mito_len_max_nm": len_mito_max_nm,
        "mito_rmin_nm": float(rmin_nm),
        "mito_rmax_nm": float(rmax_nm),
        "mito_r_circle_nm": float(r_circle_nm),
    }


# ============================================================
# 显示图像到 napari
# ============================================================
def show_image_in_napari(img, layer_name="Synthetic microtubules"):
    img = img.astype(np.float32)

    if layer_name in viewer.layers:
        layer = viewer.layers[layer_name]
        layer.data = img
        layer.contrast_limits = (
            float(img.min()),
            float(img.max()),
        )
    else:
        layer = viewer.add_image(
            img,
            name=layer_name,
            colormap="gray",
            contrast_limits=(float(img.min()), float(img.max())),
        )
    return layer


def remove_layers_by_prefix(prefixes):
    for layer in list(viewer.layers):
        if any(layer.name.startswith(prefix) for prefix in prefixes):
            viewer.layers.remove(layer)


def show_step1_result_layers(reference_image, sr_image, lr_image, seed):
    remove_layers_by_prefix(
        (
            "reference LR image",
            "Reference image",
            "Output SR",
            "Output LR level",
            "example LR",
            "Example LR",
            "Example GT",
            "Example ",
        )
    )

    display_rng = np.random.default_rng(seed + 100000)
    lr_level4 = add_lr_noise_level(lr_image, LR_NOISE_BANK[3], display_rng)
    lr_level9 = add_lr_noise_level(lr_image, LR_NOISE_BANK[8], display_rng)

    layers = [
        show_image_in_napari(reference_image, "reference LR image"),
        show_image_in_napari(sr_image, "Example GT"),
        show_image_in_napari(lr_image, "Example LR1"),
        show_image_in_napari(lr_level4, "Example LR2"),
        show_image_in_napari(lr_level9, "Example LR3"),
    ]

    viewer.grid.enabled = False
    clear_canvas_title()
    for layer in layers:
        layer.name_overlay.visible = False
        layer.visible = False
    layers[0].visible = True
    viewer.layers.selection.active = layers[0]
    viewer.reset_view()


def show_canvas_title(title):
    viewer.text_overlay.visible = True
    viewer.text_overlay.text = title
    viewer.text_overlay.position = "top_center"
    viewer.text_overlay.font_size = 18
    viewer.text_overlay.color = "white"


def clear_canvas_title():
    viewer.text_overlay.text = ""
    viewer.text_overlay.visible = False


def is_user_selected_path(path):
    path_text = str(path)
    return path_text not in ("", ".")


def read_2d_image(path):
    image = tiff.imread(str(path))
    image = np.asarray(image, dtype=np.float32)
    if image.ndim > 2:
        image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError("Only 2D images are supported.")
    return image


def list_reference_images(folder):
    folder = Path(folder)
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def preview_reference_path(path):
    path = Path(path)
    try:
        if path.is_dir():
            image_paths = list_reference_images(path)
            if not image_paths:
                napari.utils.notifications.show_error(
                    "No supported image files were found in the reference folder."
                )
                return
            image_path = image_paths[0]
        else:
            image_path = path

        image = read_2d_image(image_path)
    except Exception as exc:
        napari.utils.notifications.show_error(f"Failed to preview reference image: {exc}")
        return

    show_image_in_napari(image, layer_name="reference LR image")
    show_canvas_title("reference LR image")
    viewer.layers.selection.active = viewer.layers["reference LR image"]
    viewer.reset_view()


def sample_reference_images(image_paths, sample_count=REFERENCE_SAMPLE_COUNT):
    if len(image_paths) <= sample_count:
        return image_paths

    rng = np.random.default_rng(BASE_SEED)
    sampled = []
    for group in np.array_split(np.asarray(image_paths, dtype=object), sample_count):
        sampled.append(group[int(rng.integers(0, len(group)))])
    return sampled


def estimate_reference_resolution_from_paths(image_paths, pixel_size, progress=None):
    resolutions = []
    kc_values = []
    a0_values = []

    for index, image_path in enumerate(image_paths, start=1):
        if progress is not None:
            progress.setLabelText(f"Estimating reference image resolution... {index}/{len(image_paths)}")
            progress.setValue(index - 1)
            QApplication.processEvents()
            if progress.wasCanceled():
                raise RuntimeError("canceled")

        image = read_2d_image(image_path)
        resolution, kc, a0 = resolution_estimation(image, pixel_size)
        if np.isfinite(resolution):
            resolutions.append(resolution)
            kc_values.append(kc)
            a0_values.append(a0)

    if not resolutions:
        raise ValueError("No valid 2D reference images were found.")

    return (
        float(np.mean(resolutions)),
        float(np.mean(kc_values)),
        float(np.mean(a0_values)),
        read_2d_image(image_paths[0]),
        f"{len(resolutions)} image(s)",
        resolutions,
        image_paths,
    )


# ============================================================
# 每张图随机采样一组参数
# ============================================================
def sample_random_parameters(rng):
    numMicrotubules = int(
        rng.integers(
            PARAM_RANGES["numMicrotubules"][0],
            PARAM_RANGES["numMicrotubules"][1] + 1,
        )
    )

    density_factor = float(
        rng.uniform(
            PARAM_RANGES["density_factor"][0],
            PARAM_RANGES["density_factor"][1],
        )
    )

    centerfactor = float(
        rng.uniform(
            PARAM_RANGES["centerfactor"][0],
            PARAM_RANGES["centerfactor"][1],
        )
    )

    maxLength_nm = float(
        rng.uniform(
            PARAM_RANGES["maxLength_nm"][0],
            PARAM_RANGES["maxLength_nm"][1],
        )
    )

    bendStrengthBase = float(
        rng.uniform(
            PARAM_RANGES["bendStrengthBase"][0],
            PARAM_RANGES["bendStrengthBase"][1],
        )
    )

    bendStrengthEdge = float(
        rng.uniform(
            PARAM_RANGES["bendStrengthEdge"][0],
            PARAM_RANGES["bendStrengthEdge"][1],
        )
    )

    noiseLevel = float(
        rng.uniform(
            PARAM_RANGES["noiseLevel"][0],
            PARAM_RANGES["noiseLevel"][1],
        )
    )

    numClusters = int(
        rng.integers(
            PARAM_RANGES["numClusters"][0],
            PARAM_RANGES["numClusters"][1] + 1,
        )
    )

    controlmin = int(
        rng.integers(
            PARAM_RANGES["controlmin"][0],
            PARAM_RANGES["controlmin"][1] + 1,
        )
    )

    controlmax = int(
        rng.integers(
            PARAM_RANGES["controlmax"][0],
            PARAM_RANGES["controlmax"][1] + 1,
        )
    )

    if controlmax < controlmin:
        controlmax = controlmin

    addORoverlap = float(
        rng.uniform(
            PARAM_RANGES["addORoverlap"][0],
            PARAM_RANGES["addORoverlap"][1],
        )
    )

    return {
        "numMicrotubules": numMicrotubules,
        "density_factor": density_factor,
        "centerfactor": centerfactor,
        "maxLength_nm": maxLength_nm,
        "bendStrengthBase": bendStrengthBase,
        "bendStrengthEdge": bendStrengthEdge,
        "noiseLevel": noiseLevel,
        "numClusters": numClusters,
        "controlmin": controlmin,
        "controlmax": controlmax,
        "addORoverlap": addORoverlap,
    }


def get_selected_organelle_type():
    if ORGANELLE_COMBO is None:
        return ORGANELLE_TYPES[0]
    return ORGANELLE_COMBO.currentText()


def sample_organelle_parameters(organelle_type, rng):
    if organelle_type == "MT":
        return sample_random_parameters(rng)
    if organelle_type in MITO_ORGANELLE_TYPES:
        return sample_mito_parameters(rng)
    if organelle_type == "CCP":
        return dict(CCP_DEFAULT_PARAMS)
    if organelle_type == "NPCs":
        pair_dist_min = float(
            rng.uniform(NPC_PARAM_RANGES["pairDist_nm_min"][0], NPC_PARAM_RANGES["pairDist_nm_min"][1])
        )
        pair_dist_max = float(
            rng.uniform(
                max(pair_dist_min, NPC_PARAM_RANGES["pairDist_nm_max"][0]),
                NPC_PARAM_RANGES["pairDist_nm_max"][1],
            )
        )
        return {
            **NPC_DEFAULT_PARAMS,
            "numNPC": int(rng.integers(NPC_PARAM_RANGES["numNPC"][0], NPC_PARAM_RANGES["numNPC"][1] + 1)),
            "pair_ratio": float(rng.uniform(NPC_PARAM_RANGES["pair_ratio"][0], NPC_PARAM_RANGES["pair_ratio"][1])),
            "pairDist_nm_min": pair_dist_min,
            "pairDist_nm_max": pair_dist_max,
        }
    if organelle_type == "ER":
        diam_min_nm = float(rng.uniform(ER_PARAM_RANGES["diam_min_nm"][0], ER_PARAM_RANGES["diam_min_nm"][1]))
        diam_max_nm = float(
            rng.uniform(max(diam_min_nm, ER_PARAM_RANGES["diam_max_nm"][0]), ER_PARAM_RANGES["diam_max_nm"][1])
        )
        return {
            "er_preset": "sheetlow",
            "er_diam_min_nm": diam_min_nm,
            "er_diam_max_nm": diam_max_nm,
            "er_density": float(rng.uniform(ER_PARAM_RANGES["density"][0], ER_PARAM_RANGES["density"][1])),
            "er_noise_level": float(
                rng.uniform(ER_PARAM_RANGES["noise_level"][0], ER_PARAM_RANGES["noise_level"][1])
            ),
            "er_holes_ratio": float(
                rng.uniform(ER_PARAM_RANGES["holes_ratio"][0], ER_PARAM_RANGES["holes_ratio"][1])
            ),
        }
    raise NotImplementedError(f"{organelle_type} generator is not implemented yet.")


def generate_organelle_image(organelle_type, pixelsize, img_size, params, seed):
    if organelle_type == "MT":
        return generate_mt(
            pixelsize=pixelsize,
            imgSize=img_size,
            numMicrotubules=params["numMicrotubules"],
            density_factor=params["density_factor"],
            centerfactor=params["centerfactor"],
            maxLength_nm=params["maxLength_nm"],
            bendStrengthBase=params["bendStrengthBase"],
            bendStrengthEdge=params["bendStrengthEdge"],
            noiseLevel=params["noiseLevel"],
            numClusters=params["numClusters"],
            controlmin=params["controlmin"],
            controlmax=params["controlmax"],
            addORoverlap=params["addORoverlap"],
            seed=seed,
        ).astype(np.float32)

    if organelle_type == "CCP":
        return generate_ccp(
            pixelsize=pixelsize,
            imgSize=img_size,
            diam_min_nm=params["diam_min_nm"],
            diam_max_nm=params["diam_max_nm"],
            seed=seed,
        ).astype(np.float32)

    if organelle_type == "NPCs":
        return generate_npc(
            pixelsize=pixelsize,
            imgSize=img_size,
            numNPC=params["numNPC"],
            Nup_dia_nm=params["Nup_dia_nm"],
            minDist_nm=params["minDist_nm"],
            pair_ratio=params["pair_ratio"],
            pairDist_nm_min=params["pairDist_nm_min"],
            pairDist_nm_max=params["pairDist_nm_max"],
            R_nm_max=params["R_nm_max"],
            R_nm_min=params["R_nm_min"],
            ring_thick_nm=params["ring_thick_nm"],
            nSym=params["nSym"],
            ptsPerSubunit=params["ptsPerSubunit"],
            dtheta_deg=params["dtheta_deg"],
            sigma_r_nm=params["sigma_r_nm"],
            npc_gain_range=[params["npc_gain_min"], params["npc_gain_max"]],
            dot_sigma_nm=params["dot_sigma_nm"],
            seed=seed,
        ).astype(np.float32)

    if organelle_type == "ER":
        return generate_er(
            pixel_size=pixelsize,
            imgSize=img_size,
            diam_min_nm=params["er_diam_min_nm"],
            diam_max_nm=params["er_diam_max_nm"],
            density=params["er_density"],
            noise_level=params["er_noise_level"],
            holes_ratio=params["er_holes_ratio"],
            seed=seed,
        ).astype(np.float32)

    if organelle_type in MITO_ORGANELLE_TYPES:
        return generate_mito(
            pixel_size=pixelsize,
            imgSize=img_size,
            zsize=params["mito_zsize"],
            num_mito=params["mito_num_mito"],
            num_circle=params["mito_num_circle"],
            len_mito_min_nm=params["mito_len_min_nm"],
            len_mito_max_nm=params["mito_len_max_nm"],
            rmin_nm=params["mito_rmin_nm"],
            rmax_nm=params["mito_rmax_nm"],
            r_circle_nm=params["mito_r_circle_nm"],
            same_z_p=params["mito_same_z_p"],
            seed=seed,
        ).astype(np.float32)

    raise NotImplementedError(f"{organelle_type} generator is not implemented yet.")


def generate_omm_volume(pixelsize, img_size, params, seed):
    return generate_mito_volume(
        pixel_size=pixelsize,
        imgSize=img_size,
        zsize=params["mito_zsize"],
        num_mito=params["mito_num_mito"],
        num_circle=params["mito_num_circle"],
        len_mito_min_nm=params["mito_len_min_nm"],
        len_mito_max_nm=params["mito_len_max_nm"],
        rmin_nm=params["mito_rmin_nm"],
        rmax_nm=params["mito_rmax_nm"],
        r_circle_nm=params["mito_r_circle_nm"],
        same_z_p=params["mito_same_z_p"],
        seed=seed,
    ).astype(np.float32)


def sample_omm_sigma_z(rng):
    return float(OMM_SIGMA_Z_MIN + rng.random() * OMM_SIGMA_Z_SPAN)


def omm_center_z(volume):
    return max(int(volume.shape[0]) // 2 - 1, 0)


def omm_downsample_factor(lr_pixelsize, sr_pixelsize):
    factor = float(lr_pixelsize) / float(sr_pixelsize)
    if factor < 1:
        raise ValueError("For OMM, SR pixel size must be less than or equal to LR/reference pixel size.")
    return factor


def generate_omm_sr_from_volume(volume, sigma_xy, sigma_z):
    psf = auto_psf_3d(
        OMM_PSF_XY_SIZE,
        max(1, int(volume.shape[0]) // 2),
        OMM_PSF_SHIFT,
        sigma_xy,
        sigma_z,
        OMM_PSF_SCATTER_GAIN,
    ).astype(np.float32, copy=False)
    gt = convolve_volume_center_slice(volume, psf, center_z=omm_center_z(volume))
    return normalize_unit_float(gt), psf


def generate_omm_lr_sr_images(params, seed, config, rng):
    volume = generate_omm_volume(config["sr_pixelsize"], IMG_SIZE, params, seed)
    sigma_z = sample_omm_sigma_z(rng)
    sr_img, _psf = generate_omm_sr_from_volume(volume, config["omm_sr_sigma_xy"], sigma_z)
    downsample_factor = config.get("omm_lr_downsample_factor") or omm_downsample_factor(
        config["lr_pixelsize"],
        config["sr_pixelsize"],
    )
    lr_emitter = downsample_image(sr_img, factor=downsample_factor)
    lr_img = convolve_same(lr_emitter, config["lr_psf_crop"]).astype(np.float32)
    sr_emitter = normalize_unit_float(volume[omm_center_z(volume)])
    return lr_emitter, sr_emitter, lr_img, sr_img, sigma_z


# ============================================================
# 根据一组参数生成 LR/SR emitter figure
# ============================================================
def generate_lr_sr_emitters(params, seed, config):
    organelle_type = config["organelle_type"]
    if config["two_x_upsamp"]:
        emitter = generate_organelle_image(organelle_type, config["sr_pixelsize"], IMG_SIZE, params, seed)
        return downsample_image(emitter, factor=2), emitter

    lr_emitter = generate_organelle_image(organelle_type, config["lr_pixelsize"], IMG_SIZE, params, seed)
    sr_emitter = generate_organelle_image(organelle_type, config["sr_pixelsize"], IMG_SIZE, params, seed)
    return lr_emitter, sr_emitter


def normalize_unit_float(image):
    image = np.asarray(image, dtype=np.float32)
    image = image - float(np.min(image))
    max_value = float(np.max(image))
    if max_value > 0:
        image = image / max_value
    return image.astype(np.float32, copy=False)


def normalize_by_positive_max(noise):
    noise = np.asarray(noise, dtype=np.float32)
    max_value = float(np.max(noise))
    if max_value > 0:
        return noise / max_value

    abs_max = float(np.max(np.abs(noise)))
    if abs_max > 0:
        return noise / abs_max
    return noise


def add_lr_noise_level(image, noise_level, rng):
    wf = normalize_unit_float(image)
    gaussian_noise = normalize_by_positive_max(rng.normal(size=wf.shape).astype(np.float32))

    poisson_counts = rng.poisson(np.clip(wf, 0.0, None) * POISSON_COUNT_LEVEL)
    poisson_image = poisson_counts.astype(np.float32) / POISSON_COUNT_LEVEL
    poisson_noise = normalize_by_positive_max(poisson_image - wf)

    noisy = wf + float(noise_level) * gaussian_noise + float(noise_level) * poisson_noise
    return normalize_unit_float(noisy)


def parse_train_val_ratio(ratio_text):
    text = str(ratio_text).strip()
    for separator in (":", "：", "/", ","):
        if separator in text:
            parts = [part.strip() for part in text.split(separator)]
            break
    else:
        raise ValueError("Use a train:val ratio such as 8:2.")

    if len(parts) != 2:
        raise ValueError("Use a train:val ratio such as 8:2.")

    train_ratio = float(parts[0])
    val_ratio = float(parts[1])
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("Train and val ratio values must be greater than 0.")
    return train_ratio, val_ratio


def build_split_names(n_images, train_ratio, val_ratio):
    train_count = int(round(n_images * train_ratio / (train_ratio + val_ratio)))
    train_count = min(max(train_count, 1), n_images)
    if n_images > 1 and train_count == n_images:
        train_count = n_images - 1
    return ["train" if index < train_count else "val" for index in range(n_images)]


# ============================================================
# 批量生成 LR/SR 数据集
# ============================================================
@magicgui(
    call_button="Step 2: Generate LR/SR Dataset",
    out_dir={"mode": "d", "label": "Out dir"},
    prefix={"label": "Prefix"},
    n_images={"widget_type": "SpinBox", "label": "N base images", "min": 1, "max": 100000, "step": 1},
    save_emitter_maps={
        "widget_type": "CheckBox",
        "label": "Save emitter maps",
    },
    train_val_ratio={
        "widget_type": "LineEdit",
        "label": "Train:val ratio",
    },
)
def batch_generate_dataset_widget(
    out_dir: Path = DEFAULT_OUTPUT_DIR / "mt_dataset_lr_sr",
    prefix: str = "mt",
    n_images: int = 100,
    save_emitter_maps: bool = False,
    train_val_ratio: str = "8:2",
):
    if LAST_LR_SR_CONFIG is None:
        napari.utils.notifications.show_error(
            "Please run Step 1 first to estimate resolution and fit LR/SR PSF parameters."
        )
        return

    config = LAST_LR_SR_CONFIG
    try:
        train_ratio_value, val_ratio_value = parse_train_val_ratio(train_val_ratio)
    except ValueError as exc:
        napari.utils.notifications.show_error(str(exc))
        return

    split_names = build_split_names(n_images, train_ratio_value, val_ratio_value)
    out_dir = Path(out_dir)
    split_dirs = {}
    for split_name in ("train", "val"):
        split_dirs[split_name] = {
            "lr": out_dir / split_name / "LR",
            "sr": out_dir / split_name / "SR",
        }
        split_dirs[split_name]["lr"].mkdir(parents=True, exist_ok=True)
        split_dirs[split_name]["sr"].mkdir(parents=True, exist_ok=True)
        if save_emitter_maps:
            split_dirs[split_name]["lr_emitter"] = out_dir / split_name / "emitter_LR"
            split_dirs[split_name]["sr_emitter"] = out_dir / split_name / "emitter_SR"
            split_dirs[split_name]["lr_emitter"].mkdir(parents=True, exist_ok=True)
            split_dirs[split_name]["sr_emitter"].mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(BASE_SEED)
    csv_path = out_dir / "parameters.csv"

    fieldnames = [
        "filename",
        "split",
        "base_index",
        "noise_index",
        "lr_noise_level",
        "emitter_lr_filename",
        "emitter_sr_filename",
        "organelle_type",
        "seed",
        "lr_pixelsize",
        "sr_pixelsize",
        "lr_lambda",
        "sr_lambda",
        "lr_target_resolution",
        "sr_target_resolution",
        "lr_measured_resolution",
        "sr_measured_resolution",
        "two_x_upsamp",
        "lr_imgSize",
        "sr_imgSize",
        "numMicrotubules",
        "density_factor",
        "centerfactor",
        "maxLength_nm",
        "bendStrengthBase",
        "bendStrengthEdge",
        "noiseLevel",
        "numClusters",
        "controlmin",
        "controlmax",
        "addORoverlap",
        "diam_min_nm",
        "diam_max_nm",
        "numNPC",
        "Nup_dia_nm",
        "minDist_nm",
        "pair_ratio",
        "pairDist_nm_min",
        "pairDist_nm_max",
        "R_nm_min",
        "R_nm_max",
        "ring_thick_nm",
        "nSym",
        "ptsPerSubunit",
        "dtheta_deg",
        "sigma_r_nm",
        "npc_gain_min",
        "npc_gain_max",
        "dot_sigma_nm",
        "mito_zsize",
        "mito_num_p",
        "mito_num_total",
        "mito_long_ratio",
        "mito_num_mito",
        "mito_num_circle",
        "mito_same_z_p",
        "mito_len_p",
        "mito_len_min_nm",
        "mito_len_max_nm",
        "mito_rmin_nm",
        "mito_rmax_nm",
        "mito_r_circle_nm",
        "omm_sr_sigma_xy",
        "omm_sr_sigma_z",
        "omm_lr_downsample_factor",
        "er_preset",
        "er_diam_min_nm",
        "er_diam_max_nm",
        "er_density",
        "er_noise_level",
        "er_holes_ratio",
    ]

    generated_count = 0
    total_pairs = n_images * len(LR_NOISE_BANK)
    is_omm_config = config["organelle_type"] in MITO_ORGANELLE_TYPES
    lr_psf_crop = config.get("lr_psf_crop")
    sr_psf_crop = config.get("sr_psf_crop")
    if lr_psf_crop is None:
        lr_psf_crop = psf_cropper(
            generate_psf(
                config["lr_img_size"],
                config["lr_pixelsize"],
                config["lr_lambda"],
                config["NA"],
                config["refractive_index"],
            )
        )
    config["lr_psf_crop"] = lr_psf_crop
    if not is_omm_config and sr_psf_crop is None:
        sr_psf_crop = psf_cropper(
            generate_psf(
                config["sr_img_size"],
                config["sr_pixelsize"],
                config["sr_lambda"],
                config["NA"],
                config["refractive_index"],
            )
        )
    if not is_omm_config:
        config["sr_psf_crop"] = sr_psf_crop

    progress = QProgressDialog(
        "Generating LR/SR image pairs...",
        "Cancel",
        0,
        total_pairs,
        viewer.window.qt_viewer,
    )
    progress.setWindowTitle("LR/SR Dataset Generation Progress")
    progress.setMinimumDuration(0)
    progress.setValue(0)
    QApplication.processEvents()

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(n_images):
            if progress.wasCanceled():
                break

            seed_i = BASE_SEED + i
            try:
                params = sample_organelle_parameters(config["organelle_type"], rng)
                if is_omm_config:
                    lr_emitter, sr_emitter, lr_img, sr_img, omm_sigma_z = generate_omm_lr_sr_images(
                        params,
                        seed_i,
                        config,
                        rng,
                    )
                else:
                    lr_emitter, sr_emitter = generate_lr_sr_emitters(params, seed_i, config)
                    lr_img = convolve_same(lr_emitter, lr_psf_crop).astype(np.float32)
                    sr_img = convolve_same(sr_emitter, sr_psf_crop).astype(np.float32)
                    omm_sigma_z = ""
            except NotImplementedError as exc:
                progress.close()
                napari.utils.notifications.show_error(str(exc))
                return

            split_name = split_names[i]
            emitter_lr_filename = ""
            emitter_sr_filename = ""

            if save_emitter_maps:
                emitter_lr_filename = f"{prefix}_base_{i + 1:0{DIGITS}d}_emitter_LR.tif"
                emitter_sr_filename = f"{prefix}_base_{i + 1:0{DIGITS}d}_emitter_SR.tif"
                save_float_tiff(split_dirs[split_name]["lr_emitter"] / emitter_lr_filename, lr_emitter)
                save_float_tiff(split_dirs[split_name]["sr_emitter"] / emitter_sr_filename, sr_emitter)

            for noise_index, noise_level in enumerate(LR_NOISE_BANK, start=1):
                if progress.wasCanceled():
                    break

                pair_index = i * len(LR_NOISE_BANK) + noise_index
                filename = f"{prefix}_{pair_index:0{DIGITS}d}.tif"
                lr_save_path = split_dirs[split_name]["lr"] / filename
                sr_save_path = split_dirs[split_name]["sr"] / filename
                noisy_lr_img = add_lr_noise_level(lr_img, noise_level, rng)

                save_float_tiff(lr_save_path, noisy_lr_img)
                save_float_tiff(sr_save_path, sr_img)

                writer.writerow(
                    {
                        "filename": filename,
                        "split": split_name,
                        "base_index": i + 1,
                        "noise_index": noise_index,
                        "lr_noise_level": noise_level,
                        "emitter_lr_filename": emitter_lr_filename,
                        "emitter_sr_filename": emitter_sr_filename,
                        "organelle_type": config["organelle_type"],
                        "seed": seed_i,
                        "lr_pixelsize": config["lr_pixelsize"],
                        "sr_pixelsize": config["sr_pixelsize"],
                        "lr_lambda": config["lr_lambda"],
                        "sr_lambda": config.get("sr_lambda") or "",
                        "lr_target_resolution": config["lr_target_resolution"],
                        "sr_target_resolution": config["sr_target_resolution"],
                        "lr_measured_resolution": config["lr_measured_resolution"],
                        "sr_measured_resolution": config["sr_measured_resolution"],
                        "two_x_upsamp": config["two_x_upsamp"],
                        "lr_imgSize": noisy_lr_img.shape[0],
                        "sr_imgSize": sr_img.shape[0],
                        "omm_sr_sigma_xy": config.get("omm_sr_sigma_xy", ""),
                        "omm_sr_sigma_z": omm_sigma_z,
                        "omm_lr_downsample_factor": config.get("omm_lr_downsample_factor", ""),
                        **params,
                    }
                )

                generated_count += 1

                progress.setLabelText(
                    f"Generated {generated_count}/{total_pairs} pairs: {filename}"
                )
                progress.setValue(generated_count)
                QApplication.processEvents()

            if (i + 1) % 10 == 0 or (i + 1) == n_images:
                param_text = ", ".join(f"{key}={value:.3g}" if isinstance(value, float) else f"{key}={value}" for key, value in params.items())
                print(
                    f"Saved base image {i + 1}/{n_images}: {len(LR_NOISE_BANK)} noisy LR pairs | "
                    f"{config['organelle_type']} | {param_text}"
                )

            if progress.wasCanceled():
                break

    progress.close()

    for layer in list(viewer.layers):
        if layer.name.startswith("Dataset preview "):
            viewer.layers.remove(layer)

    if generated_count == total_pairs:
        print("LR/SR dataset generation finished.")
        train_pairs = split_names.count("train") * len(LR_NOISE_BANK)
        val_pairs = split_names.count("val") * len(LR_NOISE_BANK)
        QMessageBox.information(
            viewer.window.qt_viewer,
            "Save Complete",
            f"Saved to\n{out_dir.resolve()}\n\nTrain pairs: {train_pairs}\nVal pairs: {val_pairs}",
        )
    else:
        print(f"LR/SR dataset generation canceled after {generated_count}/{total_pairs} pairs.")
    print(f"Dataset saved to: {out_dir}")
    print(f"Train base images: {split_names.count('train')}, val base images: {split_names.count('val')}")
    print(f"Save emitter maps: {save_emitter_maps}")
    print(f"Parameter table saved to: {csv_path}")


@magicgui(
    call_button="Step 1: Estimate Resolution + Fit PSF",
    out_dir={"mode": "d", "label": "Example out dir"},
    prefix={"label": "Prefix"},
    target_pixelsize={
        "widget_type": "LineEdit",
        "label": "Reference/LR pixel size (nm)",
    },
    two_x_upsamp={
        "widget_type": "CheckBox",
        "label": "2x Upsamp",
    },
    sr_pixelsize={
        "widget_type": "FloatSpinBox",
        "label": "SR pixel size (nm)",
        "min": 0,
        "max": 500,
        "step": 1,
    },
    NA={"widget_type": "FloatSpinBox", "min": 0.1, "max": 2.0, "step": 0.01},
)
def fit_psf_from_reference_widget(
    out_dir: Path = DEFAULT_OUTPUT_DIR / "mt_psf_fit",
    prefix: str = "mt_fit",
    target_pixelsize: str = "",
    two_x_upsamp: bool = False,
    sr_pixelsize: float = 0.0,
    NA: float = 1.4,
):
    global LAST_LR_SR_CONFIG

    clear_canvas_title()
    viewer.grid.enabled = False

    reference_source = None
    reference_sample_paths = []
    reference_resolutions = []
    reference_path_text = REFERENCE_PATH_EDIT.text().strip() if REFERENCE_PATH_EDIT is not None else ""
    reference_path = Path(reference_path_text) if reference_path_text else Path("")

    try:
        target_pixelsize = float(target_pixelsize)
    except ValueError:
        napari.utils.notifications.show_error("Please enter reference/LR pixel size (nm).")
        return
    if target_pixelsize <= 0:
        napari.utils.notifications.show_error("reference/LR pixel size (nm) must be greater than 0.")
        return

    if two_x_upsamp:
        sr_pixelsize = target_pixelsize / 2
        fit_psf_from_reference_widget.sr_pixelsize.value = sr_pixelsize
    elif sr_pixelsize <= 0:
        napari.utils.notifications.show_error("Please enter SR pixel size (nm), or enable 2x upsamp to set it automatically.")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lambda_min = DEFAULT_LAMBDA_MIN
    lambda_max = DEFAULT_LAMBDA_MAX
    refractive_index = DEFAULT_REFRACTIVE_INDEX
    max_iter = DEFAULT_MAX_ITER
    seed = int(np.random.default_rng().integers(1, np.iinfo(np.uint32).max))
    lr_pixelsize = target_pixelsize
    lr_target_resolution = None
    sr_target_resolution = None
    reference_step_count = 1

    progress = QProgressDialog(
        "Estimating reference image resolution...",
        "Cancel",
        0,
        reference_step_count + 2 * max_iter + 8,
        viewer.window.qt_viewer,
    )
    progress.setWindowTitle("Resolution Matching and Convolution")
    progress.setMinimumDuration(0)
    progress.setValue(0)
    QApplication.processEvents()

    active_layer = viewer.layers.selection.active
    if is_user_selected_path(reference_path) and reference_path.is_dir():
        image_paths = list_reference_images(reference_path)
        if not image_paths:
            progress.close()
            napari.utils.notifications.show_error("No supported image files were found in the reference folder.")
            return

        sampled_paths = sample_reference_images(image_paths)
        reference_step_count = len(sampled_paths)
        progress.setMaximum(len(sampled_paths) + 2 * max_iter + 8)
        try:
            (
                target_resolution,
                target_kc,
                target_a0,
                reference_image,
                reference_source,
                reference_resolutions,
                reference_sample_paths,
            ) = estimate_reference_resolution_from_paths(sampled_paths, target_pixelsize, progress)
        except RuntimeError as exc:
            progress.close()
            if str(exc) != "canceled":
                raise
            return
        except Exception as exc:
            progress.close()
            napari.utils.notifications.show_error(f"Failed to estimate resolution from reference folder: {exc}")
            return
        reference_source = f"folder: {reference_path} ({reference_source})"
    elif is_user_selected_path(reference_path) and reference_path.is_file():
        try:
            reference_image = read_2d_image(reference_path)
        except Exception as exc:
            progress.close()
            napari.utils.notifications.show_error(f"Failed to read reference image: {exc}")
            return

        target_resolution, target_kc, target_a0 = resolution_estimation(
            reference_image,
            target_pixelsize,
        )
        reference_source = f"image: {reference_path}"
        reference_sample_paths = [reference_path]
        reference_resolutions = [target_resolution]
    elif isinstance(active_layer, Image):
        reference_image = np.asarray(active_layer.data, dtype=np.float32)
        if reference_image.ndim > 2:
            reference_image = np.squeeze(reference_image)
        if reference_image.ndim != 2:
            progress.close()
            napari.utils.notifications.show_error("Only 2D images are currently supported for resolution estimation.")
            return

        target_resolution, target_kc, target_a0 = resolution_estimation(
            reference_image,
            target_pixelsize,
        )
        reference_source = f"active layer: {active_layer.name}"
    else:
        progress.close()
        napari.utils.notifications.show_warning(
            "Please select a reference file/folder, or select a dragged image layer in napari."
        )
        return

    if not np.isfinite(target_resolution):
        progress.close()
        napari.utils.notifications.show_error("Reference image resolution estimation failed. Please check the image content or pixel size.")
        return

    lr_target_resolution = target_resolution
    sr_target_resolution = target_resolution / 2

    progress.setValue(reference_step_count)
    progress.setLabelText(
        f"LR target resolution is about {lr_target_resolution:.3f} nm. Generating LR/SR emitter images..."
    )
    QApplication.processEvents()
    if progress.wasCanceled():
        progress.close()
        return

    organelle_type = get_selected_organelle_type()
    is_omm_type = organelle_type in MITO_ORGANELLE_TYPES
    rng = np.random.default_rng(seed)
    try:
        params = sample_organelle_parameters(organelle_type, rng)
        if is_omm_type:
            lr_downsample_factor = omm_downsample_factor(lr_pixelsize, sr_pixelsize)
            omm_volume = generate_omm_volume(sr_pixelsize, IMG_SIZE, params, seed)
            sr_original = normalize_unit_float(omm_volume[omm_center_z(omm_volume)])
            lr_original = downsample_image(sr_original, factor=lr_downsample_factor)
            sr_sigma_z = sample_omm_sigma_z(rng)
        else:
            emitter_figure = generate_organelle_image(
                organelle_type,
                sr_pixelsize if two_x_upsamp else lr_pixelsize,
                IMG_SIZE,
                params,
                seed,
            )
            if two_x_upsamp:
                lr_original = downsample_image(emitter_figure, factor=2)
                sr_original = emitter_figure
            else:
                lr_original = emitter_figure
                sr_original = generate_organelle_image(organelle_type, sr_pixelsize, IMG_SIZE, params, seed)
            lr_downsample_factor = 2.0 if two_x_upsamp else 1.0
            sr_sigma_z = ""
    except ValueError as exc:
        progress.close()
        napari.utils.notifications.show_error(str(exc))
        return
    except NotImplementedError as exc:
        progress.close()
        napari.utils.notifications.show_error(str(exc))
        return

    progress.setValue(reference_step_count + 1)
    if is_omm_type:
        progress.setLabelText("Fitting OMM SR 3D sigma_xy...")
    else:
        progress.setLabelText("Fitting LR image resolution to match the input image...")
    QApplication.processEvents()
    if progress.wasCanceled():
        progress.close()
        return

    def update_fit_progress(offset, label):
        def _update(step, message):
            progress.setValue(min(offset + step, progress.maximum()))
            progress.setLabelText(f"{label} resolution fitting in progress")
            QApplication.processEvents()
            if progress.wasCanceled():
                raise RuntimeError("canceled")

        return _update

    def write_summary_csv(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["organelle_type", organelle_type])
            writer.writerow(["reference_source", reference_source])
            writer.writerow(["reference_sample_count", len(reference_resolutions) or 1])
            writer.writerow(["reference_lr_pixelsize_nm_per_px", target_pixelsize])
            writer.writerow(["reference_lr_resolution_nm", lr_target_resolution])
            writer.writerow(["sr_target_resolution_nm", sr_target_resolution])
            writer.writerow(["two_x_upsamp", two_x_upsamp])
            writer.writerow(["emitter_pixelsize_nm_per_px", sr_pixelsize if (two_x_upsamp or is_omm_type) else lr_pixelsize])
            writer.writerow(["lr_generated_from_emitter_downsample", two_x_upsamp or is_omm_type])
            writer.writerow(["lr_pixelsize_nm_per_px", lr_pixelsize])
            writer.writerow(["sr_pixelsize_nm_per_px", sr_pixelsize])
            writer.writerow(["lr_lambda_nm", lr_lambda])
            writer.writerow(["sr_initial_lambda_nm", "" if sr_lambda is None else lr_lambda / 2])
            writer.writerow(["sr_lambda_nm", "" if sr_lambda is None else sr_lambda])
            if is_omm_type:
                writer.writerow(["omm_sr_sigma_xy", sr_sigma_xy])
                writer.writerow(["omm_sr_sigma_z", sr_sigma_z])
                writer.writerow(["omm_lr_downsample_factor", lr_downsample_factor])
            writer.writerow(["lr_measured_resolution_nm", lr_resolution])
            writer.writerow(["sr_measured_resolution_nm", sr_resolution])
            writer.writerow(["NA", NA])
            writer.writerow(["refractive_index", refractive_index])
            writer.writerow(["resolution_tolerance_nm", DEFAULT_TOL_RES])
            writer.writerow(["seed", seed])
            if reference_sample_paths:
                writer.writerow([])
                writer.writerow(["reference_sample_path", "estimated_resolution_nm"])
                for sample_path, sample_resolution in zip(reference_sample_paths, reference_resolutions):
                    writer.writerow([sample_path, sample_resolution])
            writer.writerow([])
            writer.writerow(["parameter", "value"])
            for key, value in params.items():
                writer.writerow([key, value])

    sr_lambda = None
    sr_sigma_xy = ""
    sr_psf_crop = None
    sr_psf_3d = None
    try:
        if is_omm_type:
            sr_sigma_xy, sr_resolution, sr_conv, sr_psf_3d, sr_info = fit_sigma_xy_3d_center_slice(
                omm_volume,
                sr_target_resolution,
                OMM_SIGMA_XY_RANGE,
                sr_sigma_z,
                sr_pixelsize,
                tol_res=DEFAULT_TOL_RES,
                max_iter=max_iter,
                psf_xy_size=OMM_PSF_XY_SIZE,
                psf_shift=OMM_PSF_SHIFT,
                scatter_gain=OMM_PSF_SCATTER_GAIN,
                center_z=omm_center_z(omm_volume),
                progress_callback=update_fit_progress(reference_step_count + 1, "OMM SR sigma_xy"),
            )
            sr_conv = normalize_unit_float(sr_conv)
            lr_original = downsample_image(sr_conv, factor=lr_downsample_factor)

            progress.setValue(reference_step_count + max_iter + 4)
            progress.setLabelText("Fitting OMM LR image resolution from downsampled GT...")
            QApplication.processEvents()
            if progress.wasCanceled():
                progress.close()
                return

            lr_lambda, lr_resolution, lr_conv, lr_psf_crop, lr_info = fit_higher_lambda_directional(
                lr_original,
                lr_target_resolution,
                [lambda_min, lambda_max],
                lr_original.shape[0],
                lr_pixelsize,
                NA,
                refractive_index,
                tol_res=DEFAULT_TOL_RES,
                max_iter=max_iter,
                progress_callback=update_fit_progress(reference_step_count + max_iter + 4, "OMM LR"),
            )
            sr_psf_crop = sr_psf_3d[int(np.ceil(sr_psf_3d.shape[0] / 2.0)) - 1]
        else:
            lr_lambda, lr_resolution, lr_conv, lr_psf_crop, lr_info = fit_higher_lambda_directional(
                lr_original,
                lr_target_resolution,
                [lambda_min, lambda_max],
                lr_original.shape[0],
                lr_pixelsize,
                NA,
                refractive_index,
                tol_res=DEFAULT_TOL_RES,
                max_iter=max_iter,
                progress_callback=update_fit_progress(reference_step_count + 1, "LR"),
            )

            progress.setValue(reference_step_count + max_iter + 4)
            progress.setLabelText("Fitting SR image resolution...")
            QApplication.processEvents()
            if progress.wasCanceled():
                progress.close()
                return

            sr_lambda, sr_resolution, sr_conv, sr_psf_crop, sr_info = fit_higher_lambda_directional(
                sr_original,
                sr_target_resolution,
                [lambda_min, lambda_max],
                sr_original.shape[0],
                sr_pixelsize,
                NA,
                refractive_index,
                tol_res=DEFAULT_TOL_RES,
                max_iter=max_iter,
                initial_lambda=lr_lambda / 2,
                progress_callback=update_fit_progress(reference_step_count + max_iter + 4, "SR"),
            )
    except RuntimeError as exc:
        progress.close()
        if str(exc) != "canceled":
            raise
        return

    LAST_LR_SR_CONFIG = {
        "organelle_type": organelle_type,
        "lr_pixelsize": lr_pixelsize,
        "sr_pixelsize": sr_pixelsize,
        "lr_lambda": lr_lambda,
        "sr_lambda": sr_lambda,
        "omm_sr_sigma_xy": sr_sigma_xy,
        "omm_sr_sigma_z": sr_sigma_z,
        "omm_sigma_xy_range": OMM_SIGMA_XY_RANGE,
        "omm_psf_xy_size": OMM_PSF_XY_SIZE,
        "omm_psf_shift": OMM_PSF_SHIFT,
        "omm_psf_scatter_gain": OMM_PSF_SCATTER_GAIN,
        "omm_lr_downsample_factor": lr_downsample_factor,
        "lr_target_resolution": lr_target_resolution,
        "sr_target_resolution": sr_target_resolution,
        "lr_measured_resolution": lr_resolution,
        "sr_measured_resolution": sr_resolution,
        "two_x_upsamp": two_x_upsamp,
        "NA": NA,
        "refractive_index": refractive_index,
        "lr_img_size": lr_original.shape[0],
        "sr_img_size": sr_original.shape[0],
        "lr_psf_crop": lr_psf_crop,
        "sr_psf_crop": sr_psf_crop,
    }

    lr_original_path = out_dir / f"{prefix}_LR_original.tif"
    lr_conv_path = out_dir / f"{prefix}_LR_convolved_lambda_{lr_lambda:.2f}.tif"
    lr_psf_path = out_dir / f"{prefix}_LR_psf_lambda_{lr_lambda:.2f}.tif"
    lr_info_path = out_dir / f"{prefix}_LR_fit_info.csv"
    sr_original_path = out_dir / f"{prefix}_SR_original.tif"
    if is_omm_type:
        sr_conv_path = out_dir / f"{prefix}_SR_convolved_sigma_xy_{sr_sigma_xy:.3f}.tif"
        sr_psf_path = out_dir / f"{prefix}_SR_psf3d_center_sigma_xy_{sr_sigma_xy:.3f}.tif"
        sr_info_path = out_dir / f"{prefix}_SR_sigma_xy_fit_info.csv"
    else:
        sr_conv_path = out_dir / f"{prefix}_SR_convolved_lambda_{sr_lambda:.2f}.tif"
        sr_psf_path = out_dir / f"{prefix}_SR_psf_lambda_{sr_lambda:.2f}.tif"
        sr_info_path = out_dir / f"{prefix}_SR_fit_info.csv"
    summary_path = out_dir / f"{prefix}_LR_SR_summary.csv"

    save_float_tiff(lr_original_path, lr_original)
    save_float_tiff(lr_conv_path, lr_conv)
    save_float_tiff(lr_psf_path, lr_psf_crop)
    save_fit_info_csv(lr_info_path, lr_target_resolution, lr_lambda, lr_resolution, lr_info)
    save_float_tiff(sr_original_path, sr_original)
    save_float_tiff(sr_conv_path, sr_conv)
    save_float_tiff(sr_psf_path, sr_psf_crop)
    if is_omm_type:
        save_parameter_fit_info_csv(sr_info_path, sr_target_resolution, "sigma_xy", sr_sigma_xy, sr_resolution, sr_info)
    else:
        save_fit_info_csv(sr_info_path, sr_target_resolution, sr_lambda, sr_resolution, sr_info)
    write_summary_csv(summary_path)

    progress.setValue(progress.maximum())
    progress.setLabelText("Save complete.")
    QApplication.processEvents()
    progress.close()

    show_step1_result_layers(reference_image, sr_conv, lr_conv, seed)

    print("LR/SR resolution fitting finished.")
    print(f"Organelle type: {organelle_type}")
    print(f"Reference source: {reference_source}")
    print(f"Input/LR pixelsize: {target_pixelsize:.6g} nm/px")
    print(f"Reference/LR target resolution: {target_resolution:.6g} nm, kcMax={target_kc:.6g}, A0={target_a0:.6g}")
    print(f"SR target resolution: {sr_target_resolution:.6g} nm")
    print(f"LR lambda: {lr_lambda:.6g} nm, measured resolution: {lr_resolution:.6g} nm")
    if is_omm_type:
        print(f"OMM SR sigma_xy: {sr_sigma_xy:.6g}, sigma_z: {sr_sigma_z:.6g}, measured resolution: {sr_resolution:.6g} nm")
    else:
        print(f"SR lambda: {sr_lambda:.6g} nm, measured resolution: {sr_resolution:.6g} nm")
    print(f"LR convolved image saved to: {lr_conv_path}")
    print(f"SR convolved image saved to: {sr_conv_path}")
    print(f"Summary saved to: {summary_path}")


def sync_sr_pixelsize_from_lr(*_):
    if not fit_psf_from_reference_widget.two_x_upsamp.value:
        return

    try:
        lr_pixelsize = float(fit_psf_from_reference_widget.target_pixelsize.value)
    except ValueError:
        return

    if lr_pixelsize > 0:
        fit_psf_from_reference_widget.sr_pixelsize.value = lr_pixelsize / 2


fit_psf_from_reference_widget.two_x_upsamp.changed.connect(sync_sr_pixelsize_from_lr)
fit_psf_from_reference_widget.target_pixelsize.changed.connect(sync_sr_pixelsize_from_lr)


def style_step1_pixelsize_inputs():
    fit_psf_from_reference_widget.target_pixelsize.native.setStyleSheet(
        """
        QLineEdit {
            background-color: #46515d;
            color: #f0f0f0;
            border: 0px;
            border-radius: 4px;
            padding: 4px 8px;
        }
        """
    )


def choose_reference_file():
    path, _ = QFileDialog.getOpenFileName(
        viewer.window.qt_viewer,
        "Select reference image",
        "",
        "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp);;All files (*)",
    )
    if path and REFERENCE_PATH_EDIT is not None:
        REFERENCE_PATH_EDIT.setText(path)
        preview_reference_path(path)


def choose_reference_dir():
    path = QFileDialog.getExistingDirectory(
        viewer.window.qt_viewer,
        "Select reference folder",
        "",
    )
    if path and REFERENCE_PATH_EDIT is not None:
        REFERENCE_PATH_EDIT.setText(path)
        preview_reference_path(path)


def build_reference_path_widget():
    global REFERENCE_PATH_EDIT

    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)

    label = QLabel("Reference file/folder")
    label.setMinimumWidth(170)

    REFERENCE_PATH_EDIT = QLineEdit()
    REFERENCE_PATH_EDIT.setPlaceholderText("Select a file or folder, or use active image layer")
    REFERENCE_PATH_EDIT.setStyleSheet(
        """
        QLineEdit {
            background-color: #1f222a;
            color: #f0f0f0;
            border: 0px;
            border-radius: 4px;
            padding: 4px 8px;
        }
        """
    )

    file_button = QPushButton("File")
    file_button.clicked.connect(choose_reference_file)
    dir_button = QPushButton("Dir")
    dir_button.clicked.connect(choose_reference_dir)

    layout.addWidget(label)
    layout.addWidget(REFERENCE_PATH_EDIT, 1)
    layout.addWidget(file_button)
    layout.addWidget(dir_button)
    return widget


def clear_cached_lr_sr_config(*_):
    global LAST_LR_SR_CONFIG
    LAST_LR_SR_CONFIG = None


def build_organelle_selector_widget():
    global ORGANELLE_COMBO

    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)

    label = QLabel("Organelle type")
    label.setMinimumWidth(170)

    ORGANELLE_COMBO = QComboBox()
    ORGANELLE_COMBO.addItems(ORGANELLE_TYPES)
    ORGANELLE_COMBO.currentTextChanged.connect(clear_cached_lr_sr_config)

    layout.addWidget(label)
    layout.addWidget(ORGANELLE_COMBO, 1)
    return widget


def build_separator():
    separator = QFrame()
    separator.setFrameShape(QFrame.HLine)
    separator.setFrameShadow(QFrame.Sunken)
    return separator


def build_workflow_widget():
    style_step1_pixelsize_inputs()

    workflow_widget = QWidget()
    layout = QVBoxLayout(workflow_widget)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(8)

    layout.addWidget(build_organelle_selector_widget())
    layout.addWidget(build_separator())

    step1_label = QLabel("Step 1 - Estimate Resolution")
    step1_label.setStyleSheet("font-weight: 600;")
    layout.addWidget(step1_label)
    layout.addWidget(build_reference_path_widget())
    layout.addWidget(fit_psf_from_reference_widget.native)

    layout.addWidget(build_separator())

    step2_label = QLabel("Step 2 - Generate Dataset")
    step2_label.setStyleSheet("font-weight: 600;")
    layout.addWidget(step2_label)
    layout.addWidget(batch_generate_dataset_widget.native)
    layout.addStretch(1)

    return workflow_widget


class VoiceAppWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        global viewer, LAST_LR_SR_CONFIG, REFERENCE_PATH_EDIT, ORGANELLE_COMBO

        viewer = napari_viewer
        viewer.title = APP_NAME
        LAST_LR_SR_CONFIG = None
        REFERENCE_PATH_EDIT = None
        ORGANELLE_COMBO = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(build_workflow_widget())


if __name__ == "__main__":
    viewer = napari.Viewer(title=APP_NAME)
    viewer.window.add_dock_widget(
        VoiceAppWidget(viewer),
        area="right",
        name=APP_NAME,
    )
    napari.run()
