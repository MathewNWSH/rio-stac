"""test media type functions."""

import datetime
import json
import os
import sys
import warnings

import numpy
import pystac
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from rio_stac.stac import create_stac_item, get_raster_info

from .conftest import requires_hdf4, requires_hdf5

PREFIX = os.path.join(os.path.dirname(__file__), "fixtures")
input_date = datetime.datetime.now(datetime.timezone.utc)


@pytest.mark.parametrize(
    "file",
    [
        "dataset_nodata_nan.tif",
        "dataset_nodata_and_nan.tif",
        "dataset_cog.tif",
        "dataset_gdalcog.tif",
        "dataset_geo.tif",
        "dataset.tif",
        "dataset.jp2",
        "dataset.jpg",
        "dataset.png",
        "dataset.webp",
        "dataset_gcps.tif",
        "issue_22.tif",
        "dataset_dateline.tif",
        "dataset_int16_nodata.tif",
    ],
)
def test_create_item(file):
    """Should run without exceptions."""
    src_path = os.path.join(PREFIX, file)
    with rasterio.open(src_path) as src_dst:
        assert create_stac_item(
            src_dst, input_datetime=input_date, with_raster=True
        ).validate()


@requires_hdf4
def test_hdf4():
    """Test hdf4."""
    src_path = os.path.join(PREFIX, "dataset.hdf")
    with rasterio.open(src_path) as src_dst:
        assert create_stac_item(src_dst, input_datetime=input_date).validate()


@requires_hdf5
def test_hdf5():
    """Test hdf5."""
    src_path = os.path.join(PREFIX, "dataset.h5")
    with rasterio.open(src_path) as src_dst:
        assert create_stac_item(src_dst, input_datetime=input_date).validate()


def test_create_item_options():
    """Should return the correct mediatype."""
    src_path = os.path.join(PREFIX, "dataset_cog.tif")

    # pass string
    assert create_stac_item(src_path, input_datetime=input_date).validate()

    # default COG
    item = create_stac_item(
        src_path,
        input_datetime=input_date,
        asset_media_type=pystac.MediaType.COG,
        with_proj=False,
    )
    assert item.validate()
    item_dict = item.to_dict()
    assert item_dict["assets"]["asset"]["type"] == pystac.MediaType.COG
    assert item_dict["links"] == []
    assert item_dict["stac_extensions"] == []
    assert list(item_dict["properties"]) == ["datetime"]

    # additional extensions and properties
    item = create_stac_item(
        src_path,
        input_datetime=input_date,
        extensions=["https://stac-extensions.github.io/scientific/v1.0.0/schema.json"],
        properties={"sci:citation": "A nice image"},
        with_proj=False,
    )
    assert item.validate()
    item_dict = item.to_dict()
    assert item_dict["links"] == []
    assert item_dict["stac_extensions"] == [
        "https://stac-extensions.github.io/scientific/v1.0.0/schema.json",
    ]
    assert "datetime" in item_dict["properties"]
    assert "proj:code" not in item_dict["properties"]
    assert "sci:citation" in item_dict["properties"]

    # additional extensions and properties
    item = create_stac_item(
        src_path,
        input_datetime=input_date,
        extensions=["https://stac-extensions.github.io/scientific/v1.0.0/schema.json"],
        properties={"sci:citation": "A nice image"},
        with_proj=True,
    )
    assert item.validate()
    item_dict = item.to_dict()
    assert item_dict["links"] == []
    assert item_dict["stac_extensions"] == [
        "https://stac-extensions.github.io/scientific/v1.0.0/schema.json",
        "https://stac-extensions.github.io/projection/v2.0.0/schema.json",
    ]
    assert "datetime" in item_dict["properties"]
    assert "proj:code" in item_dict["properties"]
    assert "proj:wkt2" not in item_dict["properties"]
    assert "proj:projjson" not in item_dict["properties"]
    assert "sci:citation" in item_dict["properties"]

    # external assets
    assets = {"cog": pystac.Asset(href=src_path)}
    item = create_stac_item(
        src_path, input_datetime=input_date, assets=assets, with_proj=False
    )
    assert item.validate()
    item_dict = item.to_dict()
    assert item_dict["assets"]["cog"]
    assert item_dict["links"] == []
    assert item_dict["stac_extensions"] == []
    assert "datetime" in item_dict["properties"]

    # collection
    item = create_stac_item(
        src_path, input_datetime=input_date, collection="mycollection", with_proj=False
    )
    assert item.validate()
    item_dict = item.to_dict()
    assert item_dict["links"][0]["href"] == "mycollection"
    assert item_dict["stac_extensions"] == []
    assert "datetime" in item_dict["properties"]
    assert item_dict["collection"] == "mycollection"

    item = create_stac_item(
        src_path,
        input_datetime=input_date,
        collection="mycollection",
        collection_url="https://stac.somewhere.io/mycollection.json",
        with_proj=False,
    )
    assert item.validate()
    item_dict = item.to_dict()
    assert item_dict["links"][0]["href"] == "https://stac.somewhere.io/mycollection.json"
    assert item_dict["stac_extensions"] == []
    assert "datetime" in item_dict["properties"]
    assert item_dict["collection"] == "mycollection"


def test_proj_without_proj():
    """Use the Proj extension without proj info."""
    src_path = os.path.join(PREFIX, "dataset.tif")

    # additional extensions and properties
    item = create_stac_item(
        src_path,
        input_datetime=input_date,
        with_proj=True,
    )
    assert item.validate()
    item_dict = item.to_dict()
    assert item_dict["links"] == []
    assert item_dict["stac_extensions"] == [
        "https://stac-extensions.github.io/projection/v2.0.0/schema.json",
    ]
    assert "datetime" in item_dict["properties"]
    # Code should be set to None
    assert not item_dict["properties"]["proj:code"]
    assert item_dict["properties"]["proj:bbox"]


def test_create_item_raster():
    """Should return a valid item with raster properties."""
    src_path = os.path.join(PREFIX, "dataset_cog.tif")
    item = create_stac_item(
        src_path,
        input_datetime=input_date,
        with_raster=True,
        raster_max_size=128,
    )
    assert item.validate()
    item_dict = item.to_dict()
    assert item_dict["links"] == []
    assert item_dict["stac_extensions"] == [
        "https://stac-extensions.github.io/raster/v2.0.0/schema.json",
    ]
    assert "bands" in item_dict["assets"]["asset"]
    assert len(item_dict["assets"]["asset"]["bands"]) == 1

    # Nodata=None not in the properties
    assert "nodata" not in item_dict["assets"]["asset"]["bands"][0]

    # Unit=None not in the properties
    assert "unit" not in item_dict["assets"]["asset"]["bands"][0]

    assert item_dict["assets"]["asset"]["bands"][0]["raster:sampling"] in [
        "point",
        "area",
    ]

    src_path = os.path.join(PREFIX, "dataset_nodata_nan.tif")
    item = create_stac_item(src_path, input_datetime=input_date, with_raster=True)
    assert item.validate()
    item_dict = item.to_dict()
    assert item_dict["stac_extensions"] == [
        "https://stac-extensions.github.io/raster/v2.0.0/schema.json",
    ]
    assert "bands" in item_dict["assets"]["asset"]

    assert item_dict["assets"]["asset"]["bands"][0]["nodata"] == "nan"

    src_path = os.path.join(PREFIX, "dataset_with_offsets.tif")
    item = create_stac_item(
        src_path,
        input_datetime=input_date,
        with_raster=True,
    )
    assert item.validate()
    item_dict = item.to_dict()
    assert item_dict["stac_extensions"] == [
        "https://stac-extensions.github.io/raster/v2.0.0/schema.json",
    ]
    assert "bands" in item_dict["assets"]["asset"]
    assert item_dict["assets"]["asset"]["bands"][0]["raster:scale"] == 0.0001
    assert item_dict["assets"]["asset"]["bands"][0]["raster:offset"] == 1000.0


def test_create_item_raster_with_gcps():
    """Should return a valid item with raster properties."""
    src_path = os.path.join(PREFIX, "dataset_gcps.tif")
    item = create_stac_item(
        src_path, input_datetime=input_date, with_raster=True, with_proj=True
    )
    assert item.validate()


@pytest.mark.xfail
def test_dateline_polygon_split():
    """make sure we return a multipolygon."""
    src_path = os.path.join(PREFIX, "dataset_dateline.tif")
    item = create_stac_item(
        src_path, input_datetime=input_date, with_raster=True, with_proj=True
    )
    item_dict = item.to_dict()
    assert item_dict["geometry"]["type"] == "MultiPolygon"


def test_negative_nodata():
    """Make sure we catch valid nodata (issue 33)."""
    src_path = os.path.join(PREFIX, "dataset_int16_nodata.tif")
    item = create_stac_item(
        src_path, input_datetime=input_date, with_raster=True, with_proj=True
    )
    item_dict = item.to_dict()
    assert item_dict["assets"]["asset"]["bands"][0]["nodata"] == -9999


def test_create_item_eo():
    """Should return a valid item with eo properties."""
    src_path = os.path.join(PREFIX, "dataset_cog.tif")
    item = create_stac_item(src_path, with_eo=True)
    assert item.validate()
    item_dict = item.to_dict()
    assert item_dict["links"] == []
    assert item_dict["stac_extensions"] == [
        "https://stac-extensions.github.io/eo/v2.0.0/schema.json",
    ]
    assert "bands" in item_dict["assets"]["asset"]
    assert len(item_dict["assets"]["asset"]["bands"]) == 1
    assert item_dict["assets"]["asset"]["bands"][0]["eo:common_name"] == "pan"
    assert item_dict["assets"]["asset"]["bands"][0]["description"] == "gray"

    src_path = os.path.join(PREFIX, "dataset_description.tif")
    item = create_stac_item(src_path, with_eo=True)
    assert item.validate()
    item_dict = item.to_dict()
    assert len(item_dict["assets"]["asset"]["bands"]) == 1
    assert item_dict["assets"]["asset"]["bands"][0]["eo:common_name"] == "pan"
    assert item_dict["assets"]["asset"]["bands"][0]["description"] == "b1"

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="FALSE"):
        src_path = os.path.join(PREFIX, "dataset_cloud_date_metadata.tif")
        item = create_stac_item(src_path, with_eo=True)
    assert item.validate()
    item_dict = item.to_dict()
    assert "eo:cloud_cover" in item_dict["properties"]


def test_eo_band_imagery_metadata():
    """Band IMAGERY metadata should map to eo fields."""
    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff",
            height=1,
            width=1,
            count=1,
            dtype="uint16",
            crs="EPSG:4326",
            transform=from_origin(0, 1, 1, 1),
        ) as dst:
            dst.write(numpy.ones((1, 1, 1), dtype="uint16"))
            dst.update_tags(ns="IMAGERY", CENTRAL_WAVELENGTH_UM="0.65", FWHM_UM="0.05")

            item = create_stac_item(dst, with_eo=True)
            assert item.validate()

    band = item.to_dict()["assets"]["asset"]["bands"][0]
    assert band["eo:center_wavelength"] == 0.65
    assert band["eo:full_width_half_max"] == 0.05


def test_create_item_datetime():
    """Should return a valid item with datetime from IMD."""
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="FALSE"):
        src_path = os.path.join(PREFIX, "dataset_cloud_date_metadata.tif")
        item = create_stac_item(src_path, with_eo=True)
    assert item.validate()
    item_dict = item.to_dict()
    assert item_dict["properties"]["datetime"] == "2011-05-01T13:00:00Z"

    src_path = os.path.join(PREFIX, "dataset_tiff_datetime.tif")
    item = create_stac_item(src_path, with_eo=True)
    assert item.validate()
    item_dict = item.to_dict()
    assert item_dict["properties"]["datetime"] == "2023-10-30T11:37:13Z"


def test_densify_geom():
    """Should run without exceptions."""
    src_path = os.path.join(PREFIX, "dataset_geom.tif")

    item = create_stac_item(src_path, with_eo=True)
    assert item.validate()
    item_dict = item.to_dict()

    item_dens = create_stac_item(src_path, geom_densify_pts=21)
    assert item_dens.validate()
    item_dens_dict = item_dens.to_dict()

    assert item_dict["bbox"] != item_dens_dict["bbox"]


def test_mars_dataset():
    """Test with Mars Dataset."""
    MARS2000_SPHERE = rasterio.crs.CRS.from_proj4("+proj=longlat +R=3396190 +no_defs")
    src_path = os.path.join(PREFIX, "dataset_mars.tif")

    item = create_stac_item(src_path, geographic_crs=MARS2000_SPHERE, with_proj=True)
    assert item.validate()
    item_dict = item.to_dict()

    assert not item_dict["properties"].get("proj:code")
    assert (
        "proj:projjson" in item_dict["properties"]
        or "proj:wkt2" in item_dict["properties"]
    )


def test_json_serialization():
    """Test JSON serialization"""
    src_path = os.path.join(PREFIX, "dataset_cog.tif")
    item = create_stac_item(src_path, with_raster=True)
    assert item.validate()
    item_dict = item.to_dict()
    assert json.dumps(item_dict)


@pytest.mark.xfail(sys.version_info < (3, 10), reason="Old numpy do not raise error")
def test_stats_unique_values():
    """issue 68 -
    ref: https://github.com/developmentseed/rio-stac/issues/68
    """
    src_path = os.path.join(
        PREFIX, "S2A_MSIL2A_20220722T105631_N0400_R094_T31TCN_20220722T171159.tif"
    )
    with pytest.warns(UserWarning):
        item = create_stac_item(src_path, with_raster=True)
    assert item.validate()
    item_dict = item.to_dict()
    assert "bands" in item_dict["assets"]["asset"]
    stats = item_dict["assets"]["asset"]["bands"][9]["raster:histogram"]
    assert len(stats["buckets"]) == 3

    # Should not raise warnings when bins is good
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        item = create_stac_item(src_path, with_raster=True, histogram_bins=3)

    with pytest.raises(ValueError):
        create_stac_item(
            src_path, with_raster=True, histogram_bins=3, histogram_range=(0, -1)
        )

    with pytest.raises(ValueError):
        create_stac_item(src_path, with_raster=True, histogram_range=(0, -1))


def test_create_item_raster_custom_histogram():
    """Should return a valid item with raster properties."""
    src_path = os.path.join(PREFIX, "dataset_cog.tif")
    item = create_stac_item(
        src_path,
        input_datetime=input_date,
        with_raster=True,
        raster_max_size=128,
        histogram_bins=5,
        histogram_range=[0, 10],
    )
    assert item.validate()
    item_dict = item.to_dict()
    stats = item_dict["assets"]["asset"]["bands"][0]["raster:histogram"]
    assert len(stats["buckets"]) == 5
    assert stats["max"] == 10
    assert stats["min"] == 0


def test_stats_with_nan_missing_nodata():
    """Stats should ignore nan/inf values.

    Ref: https://github.com/developmentseed/rio-stac/issues/70
    """
    src_path = os.path.join(PREFIX, "dataset_missing_nodata_nan.tif")
    with rasterio.open(src_path) as src:
        arr = src.read(masked=True)
        assert numpy.isnan(arr.max().item())

        info = get_raster_info(src)
        assert info[0]["statistics"]["minimum"] > 0
        assert info[0]["statistics"]["maximum"] > 0
