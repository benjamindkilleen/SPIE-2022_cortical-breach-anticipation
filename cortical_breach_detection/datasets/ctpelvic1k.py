from __future__ import absolute_import
from __future__ import annotations

import logging
import re
from pathlib import Path
from shutil import rmtree
from time import sleep
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import deepdrr
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
from deepdrr import geo
from deepdrr import vis
from deepdrr.utils import data_utils
from deepdrr.utils import image_utils
from PIL import Image
from rich.progress import Progress
from torch.utils.data import DataLoader

from .. import utils
from ..console import console

log = logging.getLogger(__name__)


class CTPelvic1K:
    def __init__(
        self,
        root: str = "~/datasets",
        mode: Literal["train", "val", "test"] = "train",
        download: bool = False,
        generate: bool = False,
        overwrite: bool = False,
        progress_values: List[float] = [0.4, 0.9],
        num_trajectories: int = 10,
        num_views: int = 3,
        max_startpoint_offset: float = 8,
        max_endpoint_offset: float = 15,
        cortical_breach_threshold: float = 1.50,
        num_breach_detection_points: int = 200,
        max_alpha_error: float = 5,
        max_beta_error: float = 2,
        max_isocenter_error: Union[float, List[float]] = 10,
        split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        carm: Dict[str, Any] = {},
        projector: Dict[str, Any] = {},
    ):
        """Make the dataset.

        Args:
            root (str, optional): Where datasets are stored in general. Defaults to "~/datasets".
            download: Download the CT or X-Ray data.

            max_startpoint_offset (float, optional): How far to offset the startpoint in mm. Defaults to 8.
            max_endpoint_offset (float, optional): How far to offset the endpoint in mm. Defaults to 15.
            num_trajectories_per_sample (int, optional): Number of trajectories to randomly sample.
                The first trajectory is always the annotated one.
                Additional trajectories are randomly sampled by moving the start and endpoints. Defaults to 10.
            num_progress_values (int, optional): Number of points along the trajectory to
                place the KWire at, spaced regularly from a random start point near the entry to the endpoint. Defaults to 10.
            num_views_per_point (int, optional): Number of views to capture for each point on each trajectory,
                for each sample. The views cycle through AP, obturator oblique, and iliac oblique,
                plus some randomness. Defaults to 1.
            cortical_breach_threshold (float, optional): The density value (not the HU value!) above which peaks
                are considered a cortical breach. See `conv_hu_to_density` for the conversion.
                Defaults to 1.45.
            num_breach_detection_points: Number of points in the CT to interpolate along the trajectory for detecting the breach.
            carm (Dict[str, Any], optional): Keyword arguments to pass to the MobileCArm. Defaults to {}.
            projector (Dict[str, Any], optional): Keyword arguments to pass to the projector. Defaults to {}.
        """
        self.root = Path(root).expanduser()
        self.dataset_dir = self.root / "CTPelvic1K"
        self.mode = mode
        self.max_startpoint_offset = max_startpoint_offset
        self.max_endpoint_offset = max_endpoint_offset
        self.progress_values = np.sort(progress_values)
        self.num_trajectories = num_trajectories
        self.num_views = num_views
        self.num_progress_values = len(self.progress_values)
        self.cortical_breach_threshold = cortical_breach_threshold
        self.num_breach_detection_points = num_breach_detection_points
        self.max_alpha_error = max_alpha_error
        self.max_beta_error = max_beta_error
        self.max_isocenter_error = max_isocenter_error
        self.split = split
        self.carm = deepdrr.MobileCArm(**carm)
        self.projector_config = projector

        self.images_per_sample = self.num_views * self.num_progress_values * self.num_trajectories
        self.clinic_data_dir = self.dataset_dir / "CTPelvic1K_dataset6_data"
        self.clinic_trajectories_dir = self.dataset_dir / "CTPelvic1K_dataset6_trajectories"
        self.projections_dir = (
            self.dataset_dir
            / "CTPelvic1K_dataset6_projections_{:02d}-views_{:02d}-trajs_{:s}".format(
                self.num_views, self.num_trajectories, "-".join([f"{int(p):02d}" for p in 100 * self.progress_values])
            )
        )

        if download:
            self.download()

        self.sample_paths = self._get_sample_paths()
        if generate:
            self.generate(overwrite)

        image_paths = self.projections_dir.glob("*/*/*.png")
        self.image_paths = self._split_data(image_paths, mode, split)
        log.info(
            f"found {len(self.image_paths)} for {mode} set from {len(self.image_paths) / self.images_per_sample} samples"
        )

        self.num_negative = sum([int("nobreach" in str(p)) for p in self.image_paths])
        self.num_positive = len(self.image_paths) - self.num_negative
        self.pos_weight = self.num_negative / self.num_positive
        log.info(f"num positive, negative: {self.num_positive}, {self.num_negative} -> pos_weight: {self.pos_weight}")

    def view_volumes(self):
        ct, ct_identifier = self.get_ct(
            "/home/benjamin/datasets/CTPelvic1K/CTPelvic1K_dataset6_data/dataset6_CLINIC_0001_data.nii.gz"
        )
        carm = self.carm
        kwire = deepdrr.vol.KWire.from_example()
        # vis.show(ct, kwire, carm, full=True)

        annotation_path = "/home/benjamin/datasets/CTPelvic1K/CTPelvic1K_dataset6_trajectories/dataset6_CLINIC_0001_left_kwire_trajectory.mrk.json"
        annotation = deepdrr.LineAnnotation.from_markup(annotation_path, ct)
        for _, (view_name, view) in enumerate(self.get_views(annotation, ct)):
            carm.move_to(**view)

            vis.show(ct, kwire, carm, annotation, full=True)
            break

    def _get_sample_paths(self):
        sample_paths = []
        for ct_path in sorted(list(self.clinic_data_dir.glob("*.nii.gz"))):
            if (base := self.get_base_from_ct_path(ct_path)) is None:
                continue
            annotation_paths = self.clinic_trajectories_dir.glob(f"{base}*.mrk.json")
            annotation_paths = sorted(list(annotation_paths))
            for annotation_path in annotation_paths:
                sample_paths.append((ct_path, annotation_path))
        return sample_paths

    def _split_data(
        self,
        image_paths: List[Path],
        mode: Literal["train", "val", "test"],
        split: Tuple[float, float, float],
    ):
        image_paths = list(sorted(image_paths))
        num_samples = len(self.sample_paths)
        assert num_samples * self.images_per_sample == len(
            image_paths
        ), f"{num_samples} * {self.images_per_sample} != {len(image_paths)}. Maybe regenerate the dataset with new trajectories?"

        indices = utils.split_indices(num_samples, np.array(split))
        indices *= self.images_per_sample

        idx = dict(train=0, val=1, test=2)[mode]
        return image_paths[indices[idx] : indices[idx + 1]]

    def _check_exists(self):
        return self.clinic_data_dir.exists() and self.clinic_trajectories_dir.exists()

    def download(self):
        if self._check_exists():
            return

        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        data_utils.download(
            url="https://zenodo.org/record/4588403/files/CTPelvic1K_dataset6_data.tar.gz?download=1",
            filename="CTPelvic1K_dataset6_data.tar.gz",
            root=self.dataset_dir,
            md5="6b6121e3094cb97bc452db99dd1abf56",
            extract_name="CTPelvic1K_dataset6_data",
        )
        data_utils.download(
            url="https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/bkillee1_jh_edu/EQAG-YB8hNFKrlXe0oNbJBUBnLxoMkMFTmkKE3uVdI4nVA?e=egknIp&download=1",
            filename="CTPelvic1k_dataset6_trajectories.zip",
            root=self.dataset_dir,
            md5="49659fce3058d6ab27da9969b87c1f52",
            extract_name="CTPelvic1k_dataset6_trajectories",
        )

    def _get_image_info(self, image_path: Path):
        image_path = Path(image_path)
        return utils.load_json(image_path.parent / f"{image_path.stem}_info.json")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]

        # get the insertion depth for metrics
        image_info = self._get_image_info(image_path)
        startpoint = geo.point(image_info["startpoint"])
        endpoint = geo.point(image_info["endpoint"])
        progress = float(image_info["progress"])
        insertion_depth = progress * (endpoint - startpoint).norm()

        view_name = image_info["view_name"]
        view_label = torch.tensor(
            dict(anteroposterior=0, obturator_oblique=1, inlet=2).get(view_name, -1),
            dtype=torch.int64,
        )
        if view_label == -1:
            log.critical(f"bad view name: {view_name}")

        # get the image and cortical breach label
        image = TF.pil_to_tensor(Image.open(image_path)).to(torch.float32) / 255
        cortical_breach = image_info["cortical_breach"]
        cortical_breach_label = torch.tensor([cortical_breach], dtype=torch.int64)
        target = dict(
            cortical_breach_label=cortical_breach_label,
            view=deepdrr.utils.radians(
                np.array([image_info["alpha"], image_info["beta"]]), degrees=image_info["degrees"]
            ),
        )

        info = dict(
            insertion_depth=insertion_depth,
            view_name=view_name,
            view_label=view_label,
            will_breach_cortex=torch.tensor(cortical_breach),
            progress=torch.tensor(image_info["progress"], dtype=torch.float32),
        )

        return image, target, info

    def detect_cortical_breach(self, xs: np.ndarray, ys: np.ndarray, fractured: bool = False) -> bool:
        """Decide whether the values correspond to a cortical breach.

        Args:
            xs (np.ndarray): The progress values alone the trajectory, in [0, 1].
            ys (np.ndarray): The density values of the CT, typically in [0, 2].
            fractured (bool, optional): Whether the trajectury is along a pubic fracture. Defaults to False.

        Returns:
            float: progress value beyond which cortex has been breached

        """
        return np.any(ys >= self.cortical_breach_threshold)

    def get_base_from_ct_path(self, ct_path: Union[Path, str]) -> Optional[str]:
        """Get a base stem to use for annotation or image paths, corresponding to the CT.

        Args:
            ct_path (Path): Posixpath to the ct.

        Returns:
            Optional[str]: Either the base string, or None if parsing fails.
        """

        pattern = r"(?P<base>dataset6_CLINIC_\d+)_data\.nii\.gz"
        if (m := re.match(pattern, ct_path.name)) is None:
            return None
        else:
            return m.group("base")

    @property
    def total_images(self):
        return len(self.sample_paths) * self.num_views * self.num_trajectories * self.num_progress_values

    def get_view(self, p: geo.Point3D, alpha: float, beta: float, offset: bool = False):
        if offset:
            p = p + np.random.uniform(-self.max_isocenter_error, self.max_isocenter_error, size=3)
            alpha = alpha + np.random.uniform(-self.max_alpha_error, self.max_alpha_error)
            beta = beta + np.random.uniform(-self.max_beta_error, self.max_beta_error)
        return dict(isocenter_in_world=p, alpha=alpha, beta=beta, degrees=True)

    NUM_VIEW_TYPES = 3

    def get_views(
        self, annotation: deepdrr.LineAnnotation, ct: deepdrr.Volume
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """Get the views for the given orientation of the CT volume and a KWire.

        Args:
            annotation (deepdrr.LineAnnotation): The groundtruth annotation.

        Yields:
            Generator[Tuple[str, Dict[str, Any]], None, None]: Generator over `(view_name, view_kwargs)`.
        """
        annotation_midpoint = annotation.midpoint_in_world

        assert annotation.volume.anatomical_coordinate_system == "RAS"
        right_side = annotation.endpoint.x > 0  # assumes annotation in RAS coordinates

        assert (
            self.num_views % self.NUM_VIEW_TYPES == 0
        ), f"num_views {self.num_views} must be a multiple of {self.NUM_VIEW_TYPES}"

        # first NUM_VIEW_TYPES views are from the set angles, then start adding the randomness
        for n in range(0, self.num_views, self.NUM_VIEW_TYPES):
            # AP view, centered on the midpoint of the annotation, plus some randomness.
            yield "anteroposterior", self.get_view(annotation_midpoint, 0, -15, n > 0)
            # obturator oblique view: rotate camera 45 degs away from affected side.
            yield "obturator_oblique", self.get_view(annotation_midpoint, 30 if right_side else -30, -15, n > 0)
            yield "inlet", self.get_view(annotation_midpoint, 0, -45, n > 0)

    def _check_projections_dir(self, overwrite: bool = False) -> bool:
        if self.projections_dir.exists() and not overwrite:
            log.info(
                f"Continuing with dataset at {self.projections_dir}. Set dataset.overwrite=true to completely overwrite it."
            )
            return False

        if self.projections_dir.exists():
            log.critical(f"DELETING {self.projections_dir}! (Ctrl-C to cancel)...")
            sleep(15)
            rmtree(self.projections_dir)

        self.projections_dir.mkdir()
        return False

    def get_ct(self, identifier):
        ct = deepdrr.Volume.from_nifti(identifier, use_thresholding=False)
        ct.faceup()
        # Remove the calibration bar from the CT segmentation.
        for m in ct.materials.keys():
            ct.materials[m][:, 400:, :] = 0
        return ct, identifier

    def load_annotations(self, annotation_path, ct):
        return deepdrr.LineAnnotation.from_markup(annotation_path, ct)

    def generate(self, overwrite: bool = False):
        """Generate the dataset using deepdrr.

        Args:
            overwrite (bool, optional): Whether to overwrite existing data. Defaults to False.
        """

        if self._check_projections_dir(overwrite=overwrite):
            return

        carm = self.carm
        kwire = deepdrr.vol.KWire.from_example()
        n = 0
        num_breached = 0
        total = self.total_images

        with Progress(refresh_per_second=1) as progressbar:
            task = progressbar.add_task(f"[red]Making kwire views dataset with {total} images...", total=total)
            for ct_path, annotation_path in self.sample_paths:
                base = self.get_base_from_ct_path(ct_path) + ("_left" if "left" in str(annotation_path) else "_right")
                sample_dir = self.projections_dir / base
                if sample_dir.exists() and len(list(sample_dir.glob("*/*.png"))) == self.images_per_sample:
                    n += self.images_per_sample
                    progressbar.advance(task, self.images_per_sample)
                    continue
                elif sample_dir.exists():
                    rmtree(sample_dir)

                sample_dir.mkdir()
                fractured = "fractured" in annotation_path.name
                ct, ct_identifier = self.get_ct(ct_path)

                annotation = self.load_annotations(annotation_path, ct)
                trajectory = annotation.endpoint_in_world - annotation.startpoint_in_world
                projector = deepdrr.Projector([ct, kwire], carm=carm, **self.projector_config)

                with projector:
                    # Need to get the views first in order to perturb the trajectory in the image plane.
                    for vidx, (view_name, view) in enumerate(self.get_views(annotation, ct)):
                        carm.move_to(**view)
                        view_dir = sample_dir / f"{vidx:003d}_{view_name}"
                        view_dir.mkdir()
                        image_index = 0  # view_name -> image_index

                        for progress in self.progress_values:
                            for t in range(self.num_trajectories):
                                # Get trajectory and evaluation for cortical breach.
                                if t == 0:
                                    startpoint = annotation.startpoint_in_world
                                    endpoint = annotation.endpoint_in_world
                                else:
                                    # startpoint_offset_direction = trajectory.perpendicular(random=True)
                                    # endpoint_offset_direction = trajectory.perpendicular(random=True)
                                    # Only sampling trajectories with offset orthogonal to principle ray
                                    principle_ray = self.carm.principle_ray_in_world
                                    startpoint_offset_direction = principle_ray.cross(trajectory).hat()
                                    endpoint_offset_direction = principle_ray.cross(trajectory).hat()
                                    startpoint_offset = np.random.uniform(
                                        -self.max_startpoint_offset, self.max_startpoint_offset
                                    )
                                    endpoint_offset = np.random.uniform(
                                        -self.max_endpoint_offset, self.max_endpoint_offset
                                    )
                                    startpoint = (
                                        annotation.startpoint_in_world + startpoint_offset * startpoint_offset_direction
                                    )
                                    endpoint = (
                                        annotation.endpoint_in_world + endpoint_offset * endpoint_offset_direction
                                    )

                                # Determine if trajectory will pierce the cortex
                                progress_values_fine = np.linspace(0.05, 0.95, self.num_breach_detection_points)
                                progress_points_fine = [startpoint.lerp(endpoint, p) for p in progress_values_fine]
                                ct_values = ct.interpolate(*progress_points_fine)
                                cortical_breach = self.detect_cortical_breach(
                                    progress_values_fine, ct_values, fractured=fractured
                                )
                                log.info(f"[{base}] cortical_breach: {cortical_breach}")

                                # align the kwire
                                kwire.align(startpoint, endpoint, progress=progress)

                                # Minor data augmentation, so not all views the same.
                                view["alpha"] += np.random.uniform(-2, 2)
                                view["beta"] += np.random.uniform(-2, 2)
                                carm.move_to(**view)

                                info = dict(
                                    ct=ct_identifier,
                                    annotation=annotation_path.name,
                                    startpoint=startpoint,
                                    endpoint=endpoint,
                                    progress=progress,
                                    fractured=fractured,
                                    cortical_breach=cortical_breach,
                                    view_name=view_name,
                                    **view,
                                    trajectory_points=progress_points_fine,  # in case the breach decision should change
                                    trajectory_point_values=ct_values,
                                )
                                image = projector()

                                # Save the image.
                                breach_str = "breach" if cortical_breach else "nobreach"
                                stem = f"p-{int(100*progress):03d}_t-{t:02d}_{image_index:03d}_{breach_str}"
                                utils.save_json(view_dir / f"{stem}_info.json", info)
                                image_utils.save(view_dir / f"{stem}.png", image)

                                image_index += 1
                                n += 1
                                num_breached += int(info["cortical_breach"])
                                progressbar.advance(task)

        log.info(f"{n} images contain {num_breached} instances of cortical breach, {n - num_breached} no breach.")


class CTPelvic1KDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 12,
        persistent_workers: bool = False,
        dataset: Dict[str, Any] = {},
    ):
        super().__init__()
        self.loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
        self.dataset_kwargs = dataset
        assert "mode" not in self.dataset_kwargs

    def prepare_data(self):
        CTPelvic1K(**self.dataset_kwargs)
        self.dataset_kwargs["download"] = False
        self.dataset_kwargs["generate"] = False

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_set = CTPelvic1K(mode="train", **self.dataset_kwargs)
            self.val_set = CTPelvic1K(mode="val", **self.dataset_kwargs)
            self.dims = self.train_set[0][0].shape

        if stage == "test" or stage is None:
            self.test_set = CTPelvic1K(mode="test", **self.dataset_kwargs)
            self.dims = self.test_set[0][0].shape

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return DataLoader(self.train_set, shuffle=True, **self.loader_kwargs)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_set, **self.loader_kwargs)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_set, **self.loader_kwargs)
