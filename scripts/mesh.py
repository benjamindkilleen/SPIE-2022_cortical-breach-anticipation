"""Command line tool that takes in a segmentation and creates a vtp surface model."""
import logging
from pathlib import Path
from typing import Optional

import click
import nibabel as nib
import numpy as np
import pyvista as pv
import vtk
from rich.logging import RichHandler
from vtk.util import numpy_support as nps


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, handlers=[RichHandler()])


@click.command()
@click.argument("path", nargs=1)
@click.option("-o", "--output", required=False, nargs=1, type=str, default=None)
@click.option("-t", "--threshold", required=False, nargs=1, type=float, default=0.5)
@click.option("-T", "--upper-threshold", required=False, nargs=1, type=float, default=None)
@click.option("-l", "--label", required=False, nargs=1, type=int, default=None)
@click.option("--neg/--no-neg", default=False)
def mesh(
    path: str,
    output: Optional[str] = None,
    threshold: float = 0.5,
    upper_threshold: Optional[float] = None,
    label: Optional[int] = None,
    neg: bool = True,
):
    """Make a surface mesh from the volume at path.`

    Args:
        path (str): Path to a NifTi file.
        threshold (float, optional): Voxels above this value are considered to be the interior of the surface. Defaults to 0.5.
        label (int, optional): If provided, NIFTI file is assumed to be a segmentation, and threshold arguments are ignored.
    """
    vol = nib.load(path)
    anatomical_from_ijk = vol.affine
    R = anatomical_from_ijk[:3, :3]
    t = anatomical_from_ijk[:3, 3]
    data = vol.get_fdata()

    sign = -1 if neg else 1

    vol = vtk.vtkStructuredPoints()
    vol.SetDimensions(*data.shape[:3])
    vol.SetOrigin(
        -np.sign(R[0, 0]) * t[0],
        -np.sign(R[1, 1]) * t[1],
        -np.sign(R[2, 2]) * t[2],
    )
    vol.SetSpacing(
        -abs(R[0, 0]),
        -abs(R[1, 1]),
        -abs(R[2, 2]),
    )

    if label is not None:
        data = (data == label).astype(np.uint8)
    elif upper_threshold is None:
        data = (data > threshold).astype(np.uint8)
    else:
        data = np.logical_and(data > threshold, data <= upper_threshold).astype(np.uint8)

    scalars = nps.numpy_to_vtk(data.ravel(order="F"), deep=True)
    vol.GetPointData().SetScalars(scalars)

    log.debug("marching cubes...")
    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputData(vol)
    dmc.GenerateValues(1, 1, 1)
    dmc.ComputeGradientsOff()
    dmc.ComputeNormalsOff()
    dmc.Update()

    surface = pv.wrap(dmc.GetOutput())
    if surface.is_all_triangles():
        surface.triangulate(inplace=True)

    log.debug("postprocess...")
    surface.decimate_pro(
        0.01,
        feature_angle=60,
        splitting=False,
        preserve_topology=True,
        inplace=True,
    )

    # relaxation_factor = .16 if i in [1, 2, 3, 10] else .25
    # n_iter = 23 if i in [1, 2, 3, 10] else 30
    relaxation_factor = 0.25
    n_iter = 30

    surface.smooth(
        n_iter=n_iter,
        relaxation_factor=relaxation_factor,
        feature_angle=70,
        boundary_smoothing=False,
        inplace=True,
    )

    surface.compute_normals(inplace=True)

    path = Path(path)
    if output is None:
        output = path.parent
    output = Path(output)
    if output.is_dir():
        output = output / f"{path.stem}.vtp"

    log.info(f"writing surface to {output}...")
    surface.save(output)


if __name__ == "__main__":
    mesh()
