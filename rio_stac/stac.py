"""Create STAC Item from a rasterio dataset."""

import datetime
import math
import os
import warnings
from collections.abc import Sequence
from contextlib import ExitStack

import numpy
import pystac
import rasterio
from pystac.utils import str_to_datetime
from rasterio import transform, warp
from rasterio.features import bounds as feature_bounds
from rasterio.io import DatasetReader, DatasetWriter, MemoryFile
from rasterio.vrt import WarpedVRT

PROJECTION_EXT_VERSION = "v2.0.0"
RASTER_EXT_VERSION = "v2.0.0"
EO_EXT_VERSION = "v2.0.0"

EO_COMMON_NAME_VALUES = {
    "pan",
    "coastal",
    "blue",
    "green",
    "green05",
    "yellow",
    "red",
    "rededge",
    "rededge071",
    "rededge075",
    "rededge078",
    "nir",
    "nir08",
    "nir09",
    "cirrus",
    "swir16",
    "swir22",
    "lwir",
    "lwir11",
    "lwir12",
}

EPSG_4326 = rasterio.crs.CRS.from_epsg(4326)


def bbox_to_geom(bbox: tuple[float, float, float, float]) -> dict:
    """Return a geojson geometry from a bbox."""
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
                [bbox[0], bbox[1]],
            ]
        ],
    }


def get_dataset_geom(
    src_dst: DatasetReader | DatasetWriter | WarpedVRT | MemoryFile,
    densify_pts: int = 0,
    precision: int = -1,
    geographic_crs: rasterio.crs.CRS = EPSG_4326,
) -> dict:
    """Get Raster Footprint."""
    if densify_pts < 0:
        raise ValueError("`densify_pts` must be positive")

    if src_dst.crs is not None:
        # 1. Create Polygon from raster bounds
        geom = bbox_to_geom(src_dst.bounds)

        # 2. Densify the Polygon geometry
        if src_dst.crs != geographic_crs and densify_pts:
            # Derived from code found at
            # https://stackoverflow.com/questions/64995977/generating-equidistance-points-along-the-boundary-of-a-polygon-but-cw-ccw
            coordinates = numpy.asarray(geom["coordinates"][0])

            densified_number = len(coordinates) * densify_pts
            existing_indices = numpy.arange(0, densified_number, densify_pts)
            interp_indices = numpy.arange(existing_indices[-1] + 1)
            interp_x = numpy.interp(interp_indices, existing_indices, coordinates[:, 0])
            interp_y = numpy.interp(interp_indices, existing_indices, coordinates[:, 1])
            geom = {
                "type": "Polygon",
                "coordinates": [[(x, y) for x, y in zip(interp_x, interp_y)]],
            }

        # 3. Reproject the geometry to "epsg:4326"
        geom = warp.transform_geom(src_dst.crs, geographic_crs, geom, precision=precision)
        bbox = feature_bounds(geom)

    else:
        warnings.warn(
            "Input file doesn't have CRS information, setting geometry and bbox to (-180,-90,180,90)."
        )
        bbox = (-180.0, -90.0, 180.0, 90.0)
        geom = bbox_to_geom(bbox)

    return {"bbox": list(bbox), "footprint": geom}


def get_projection_info(
    src_dst: DatasetReader | DatasetWriter | WarpedVRT | MemoryFile,
) -> dict:
    """Get projection metadata.

    The STAC projection extension allows for three different ways to describe the coordinate reference system
    associated with a raster :
    - EPSG code
    - WKT2
    - PROJJSON

    All are optional, and they can be provided altogether as well. Therefore, as long as one can be obtained from
    the data, we add it to the returned dictionary.

    see: https://github.com/stac-extensions/projection

    """

    code = None
    if src_dst.crs is not None:
        try:
            authority, value = src_dst.crs.to_authority()
            if authority and value:
                code = f"{authority}:{value}"
        except Exception:
            # Fallback to EPSG if authority extraction fails
            code = None
        if not code and src_dst.crs.is_epsg_code:
            epsg = src_dst.crs.to_epsg()
            code = f"EPSG:{epsg}" if epsg else None

    meta = {
        "code": code,
        "geometry": bbox_to_geom(src_dst.bounds),
        "bbox": list(src_dst.bounds),
        "shape": [src_dst.height, src_dst.width],
        "transform": list(src_dst.transform),
    }

    if not code and src_dst.crs:
        # WKT2
        try:
            meta["wkt2"] = src_dst.crs.to_wkt()
        except Exception as ex:
            warnings.warn(f"Could not get WKT2 from dataset : {ex}")
            # PROJJSON
            try:
                meta["projjson"] = src_dst.crs.to_dict(projjson=True)
            except (AttributeError, TypeError) as ex:
                warnings.warn(f"Could not get PROJJSON from dataset : {ex}")

    return meta


def get_eobands_info(
    src_dst: DatasetReader | DatasetWriter | WarpedVRT | MemoryFile,
) -> list:
    """Get eo:bands metadata.

    see: https://github.com/stac-extensions/eo#item-properties-or-asset-fields

    """
    eo_bands = []

    colors = src_dst.colorinterp
    for ix in src_dst.indexes:
        band_meta = {"name": f"b{ix}"}

        descr = src_dst.descriptions[ix - 1]
        color = colors[ix - 1].name if colors else None
        imagery_tags = src_dst.tags(ix, ns="IMAGERY") or src_dst.tags(ns="IMAGERY")

        # Description metadata or Colorinterp or Nothing
        description = descr or color
        if description:
            band_meta["description"] = description

        for candidate in (descr, color):
            if not candidate:
                continue
            common_name = candidate.strip().lower().replace(" ", "")
            if common_name in {"gray", "grey"}:
                common_name = "pan"
            if common_name in EO_COMMON_NAME_VALUES:
                band_meta["eo:common_name"] = common_name
                break

        cw = imagery_tags.get("CENTRAL_WAVELENGTH_UM")
        if cw is not None:
            try:
                band_meta["eo:center_wavelength"] = float(cw)
            except ValueError:
                pass

        fwhm = imagery_tags.get("FWHM_UM")
        if fwhm is not None:
            try:
                band_meta["eo:full_width_half_max"] = float(fwhm)
            except ValueError:
                pass

        eo_bands.append(band_meta)

    return eo_bands


def _get_stats(
    arr: numpy.ma.MaskedArray,
    bins: int | str | Sequence = 10,
    range: tuple[float, float] | None = None,
) -> dict:
    """Calculate array statistics."""
    # Avoid non masked nan/inf values
    arr = numpy.ma.fix_invalid(arr, copy=True)

    stats = {
        "statistics": {
            "mean": arr.mean().item(),
            "minimum": arr.min().item(),
            "maximum": arr.max().item(),
            "stddev": arr.std().item(),
            "valid_percent": float(
                numpy.count_nonzero(~arr.mask) / float(arr.data.size) * 100
            ),
        }
    }

    try:
        sample, edges = numpy.histogram(arr[~arr.mask], bins=bins, range=range)

    except ValueError as e:
        if "Too many bins for data range." in str(e):
            _, counts = numpy.unique(arr[~arr.mask], return_counts=True)
            warnings.warn(
                f"Could not calculate the histogram, fall back to automatic bin={len(counts) + 1}.",
                UserWarning,
            )
            sample, edges = numpy.histogram(arr[~arr.mask], bins=len(counts) + 1)
        else:
            raise e

    stats["raster:histogram"] = {
        "count": len(edges),
        "min": float(edges.min()),
        "max": float(edges.max()),
        "buckets": sample.tolist(),
    }

    return stats


def get_raster_info(
    src_dst: DatasetReader | DatasetWriter | WarpedVRT | MemoryFile,
    max_size: int = 1024,
    histogram_bins: int | str | Sequence = 10,
    histogram_range: tuple[float, float] | None = None,
) -> list[dict]:
    """Get raster metadata.

    see: https://github.com/stac-extensions/raster#raster-band-object

    """
    height = src_dst.height
    width = src_dst.width
    if max_size:
        if max(width, height) > max_size:
            ratio = height / width
            if ratio > 1:
                height = max_size
                width = math.ceil(height / ratio)
            else:
                width = max_size
                height = math.ceil(width * ratio)

    meta: list[dict] = []

    area_or_point = src_dst.tags().get("AREA_OR_POINT", "").lower()

    # Missing `bits_per_sample` and `spatial_resolution`
    for band in src_dst.indexes:
        value = {
            "name": f"b{band}",
            "data_type": src_dst.dtypes[band - 1],
            "raster:scale": src_dst.scales[band - 1],
            "raster:offset": src_dst.offsets[band - 1],
        }
        if area_or_point:
            value["raster:sampling"] = area_or_point

        # If the Nodata is not set we don't forward it.
        if src_dst.nodata is not None:
            if numpy.isnan(src_dst.nodata):
                value["nodata"] = "nan"
            elif numpy.isposinf(src_dst.nodata):
                value["nodata"] = "inf"
            elif numpy.isneginf(src_dst.nodata):
                value["nodata"] = "-inf"
            else:
                value["nodata"] = src_dst.nodata

        if src_dst.units[band - 1] is not None:
            value["unit"] = src_dst.units[band - 1]

        value.update(
            _get_stats(
                src_dst.read(indexes=band, out_shape=(height, width), masked=True),
                bins=histogram_bins,
                range=histogram_range,
            )
        )
        meta.append(value)

    return meta


def get_media_type(
    src_dst: DatasetReader | DatasetWriter | WarpedVRT | MemoryFile,
) -> pystac.MediaType | None:
    """Find MediaType for a raster dataset."""
    driver = src_dst.driver

    if driver == "GTiff":
        if src_dst.crs:
            return pystac.MediaType.GEOTIFF
        else:
            return pystac.MediaType.TIFF

    elif driver in [
        "JP2ECW",
        "JP2KAK",
        "JP2LURA",
        "JP2MrSID",
        "JP2OpenJPEG",
        "JPEG2000",
    ]:
        return pystac.MediaType.JPEG2000

    elif driver in ["HDF4", "HDF4Image"]:
        return pystac.MediaType.HDF

    elif driver in ["HDF5", "HDF5Image"]:
        return pystac.MediaType.HDF5

    elif driver == "JPEG":
        return pystac.MediaType.JPEG

    elif driver == "PNG":
        return pystac.MediaType.PNG

    warnings.warn("Could not determine the media type from GDAL driver.", UserWarning)
    return None


def create_stac_item(
    source: str | DatasetReader | DatasetWriter | WarpedVRT | MemoryFile,
    input_datetime: datetime.datetime | None = None,
    extensions: list[str] | None = None,
    collection: str | None = None,
    collection_url: str | None = None,
    properties: dict | None = None,
    id: str | None = None,
    assets: dict[str, pystac.Asset] | None = None,
    asset_name: str = "asset",
    asset_roles: list[str] | None = None,
    asset_media_type: str | pystac.MediaType | None = "auto",
    asset_href: str | None = None,
    with_proj: bool = False,
    with_raster: bool = False,
    with_eo: bool = False,
    raster_max_size: int = 1024,
    geom_densify_pts: int = 0,
    geom_precision: int = -1,
    geographic_crs: rasterio.crs.CRS = EPSG_4326,
    histogram_bins: int | str | Sequence = 10,
    histogram_range: tuple[float, float] | None = None,
) -> pystac.Item:
    """Create a Stac Item.

    Args:
        source (str or opened rasterio dataset): input path or rasterio dataset.
        input_datetime (datetime.datetime, optional): datetime associated with the item.
        extensions (list of str): input list of extensions to use in the item.
        collection (str, optional): name of collection the item belongs to.
        collection_url (str, optional): Link to the STAC Collection.
        properties (dict, optional): additional properties to add in the item.
        id (str, optional): id to assign to the item (default to the source basename).
        assets (dict, optional): Assets to set in the item. If set we won't create one from the source.
        asset_name (str, optional): asset name in the Assets object.
        asset_roles (list of str, optional): list of str | list of asset's roles.
        asset_media_type (str or pystac.MediaType, optional): asset's media type.
        asset_href (str, optional): asset's URI (default to input path).
        with_proj (bool): Add the `projection` extension and properties (default to False).
        with_raster (bool): Add the `raster` extension and properties (default to False).
        with_eo (bool): Add the `eo` extension and properties (default to False).
        raster_max_size (int): Limit array size from which to get the raster statistics. Defaults to 1024.
        geom_densify_pts (int): Number of points to add to each edge to account for nonlinear edges transformation (Note: GDAL uses 21).
        geom_precision (int): If >= 0, geometry coordinates will be rounded to this number of decimal.

    Returns:
        pystac.Item: valid STAC Item.

    """
    properties = properties or {}
    extensions = extensions or []
    asset_roles = asset_roles or []

    with ExitStack() as ctx:
        if isinstance(source, (DatasetReader, DatasetWriter, WarpedVRT)):
            dataset = source
        else:
            dataset = ctx.enter_context(rasterio.open(source))

        if dataset.gcps[0]:
            src_dst = ctx.enter_context(
                WarpedVRT(
                    dataset,
                    src_crs=dataset.gcps[1],
                    src_transform=transform.from_gcps(dataset.gcps[0]),
                )
            )
        else:
            src_dst = dataset

        dataset_geom = get_dataset_geom(
            src_dst,
            densify_pts=geom_densify_pts,
            precision=geom_precision,
            geographic_crs=geographic_crs,
        )

        media_type = (
            get_media_type(dataset) if asset_media_type == "auto" else asset_media_type
        )

        if "start_datetime" not in properties and "end_datetime" not in properties:
            # Try to get datetime from https://gdal.org/user/raster_data_model.html#imagery-domain-remote-sensing
            acq_date = src_dst.get_tag_item("ACQUISITIONDATETIME", "IMAGERY")
            tiff_date = src_dst.get_tag_item("TIFFTAG_DATETIME")
            dst_date = acq_date or tiff_date
            try:
                dst_datetime = str_to_datetime(dst_date) if dst_date else None
            except ValueError as err:
                warnings.warn(f"Could not get parse date: {dst_date}: {err}")
                dst_datetime = None

            input_datetime = (
                input_datetime
                or dst_datetime
                or datetime.datetime.now(datetime.timezone.utc)
            )

        satellite_id = src_dst.get_tag_item("SATELLITEID", "IMAGERY")
        if satellite_id and "platform" not in properties:
            properties["platform"] = satellite_id

        # add projection properties
        if with_proj:
            extensions.append(
                f"https://stac-extensions.github.io/projection/{PROJECTION_EXT_VERSION}/schema.json",
            )

            properties.update({
                f"proj:{name}": value
                for name, value in get_projection_info(src_dst).items()
            })

        # add raster properties
        raster_bands: list[dict] = []
        if with_raster:
            extensions.append(
                f"https://stac-extensions.github.io/raster/{RASTER_EXT_VERSION}/schema.json",
            )

            raster_bands = get_raster_info(
                dataset,
                max_size=raster_max_size,
                histogram_bins=histogram_bins,
                histogram_range=histogram_range,
            )

        eo_bands: list[dict] = []
        if with_eo:
            extensions.append(
                f"https://stac-extensions.github.io/eo/{EO_EXT_VERSION}/schema.json",
            )

            eo_bands = get_eobands_info(src_dst)

            cloudcover = src_dst.get_tag_item("CLOUDCOVER", "IMAGERY")
            if cloudcover is not None:
                properties.update({"eo:cloud_cover": int(cloudcover)})

        bands: list[dict] = []
        if raster_bands or eo_bands:
            band_count = max(len(raster_bands), len(eo_bands))
            for idx in range(band_count):
                band: dict = {}
                if idx < len(eo_bands):
                    band.update(eo_bands[idx])
                if idx < len(raster_bands):
                    band.update(raster_bands[idx])
                band.setdefault("name", f"b{idx + 1}")
                bands.append(band)

        extensions = list(dict.fromkeys(extensions))

    # item
    item = pystac.Item(
        id=id or os.path.basename(dataset.name),
        geometry=dataset_geom["footprint"],
        bbox=dataset_geom["bbox"],
        collection=collection,
        stac_extensions=extensions,
        datetime=input_datetime,
        properties=properties,
    )

    # if we add a collection we MUST add a link
    if collection:
        item.add_link(
            pystac.Link(
                pystac.RelType.COLLECTION,
                collection_url or collection,
                media_type=pystac.MediaType.JSON,
            )
        )

    # item.assets
    if assets:
        for key, asset in assets.items():
            item.add_asset(key=key, asset=asset)

    else:
        item.add_asset(
            key=asset_name,
            asset=pystac.Asset(
                href=asset_href or dataset.name,
                media_type=media_type,
                extra_fields={"bands": bands} if bands else {},
                roles=asset_roles,
            ),
        )

    return item
