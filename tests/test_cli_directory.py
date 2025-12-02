import json
import os
import shutil
from click.testing import CliRunner
from rio_stac.scripts.cli import stac
import rasterio
import pytest

@pytest.fixture
def test_dir(tmp_path):
    """Create a temporary directory with some dummy files."""
    d = tmp_path / "data"
    d.mkdir()
    
    # Create a valid raster (DATA)
    raster_path = d / "image.tif"
    # We need a valid raster. Copy one from fixtures? 
    # Assuming we can just use the fixture path logic or creating one.
    # Since I don't want to rely on external files location assumptions too much,
    # I'll try to use the one in tests/fixtures if I can find it, or create a dummy one.
    # But for a clean test, let's try to copy from a known location if possible.
    # The environment seems to have `tests/fixtures/dataset.tif`.
    fixture_path = "tests/fixtures/dataset.tif"
    if os.path.exists(fixture_path):
        shutil.copy(fixture_path, raster_path)
    
    # Create a thumbnail (3 bands, no CRS)
    # We will mock this behavior or assume we can create such a file.
    # For now, let's just create a dummy file and assume logic works if it's robust,
    # BUT the code actually calls rasterio.open().
    # So I need real files.
    
    # Let's reuse the fixture but strip CRS for thumbnail simulation?
    # It's hard to modify TIF in place easily without gdal/rasterio.
    # I'll use the existing "dataset.jpg" from fixtures which seemed to be treated as thumb in manual test.
    thumb_path = d / "thumb.jpg"
    fixture_thumb = "tests/fixtures/dataset.jpg"
    if os.path.exists(fixture_thumb):
        shutil.copy(fixture_thumb, thumb_path)

    # Create metadata
    (d / "meta.json").write_text('{"key": "value"}')
    
    # Create ignored file
    (d / "ignored.txt").write_text("ignore me")
    
    return d

def test_cli_recursive_basic(test_dir):
    """Test basic recursive directory processing."""
    runner = CliRunner()
    result = runner.invoke(stac, [str(test_dir), "--recursive"])
    assert result.exit_code == 0
    item = json.loads(result.output)
    
    assert item["id"] == "data"
    assert "image" in item["assets"]
    assert "thumb" in item["assets"]
    assert "meta" in item["assets"]
    
    # Check roles
    assert "data" in item["assets"]["image"]["roles"]
    assert "thumbnail" in item["assets"]["thumb"]["roles"]
    assert "metadata" in item["assets"]["meta"]["roles"]

def test_cli_recursive_pattern(test_dir):
    """Test recursive processing with pattern filtering."""
    runner = CliRunner()
    # Only select .tif
    result = runner.invoke(stac, [str(test_dir), "--recursive", "--pattern", "*.tif"])
    assert result.exit_code == 0
    item = json.loads(result.output)
    
    assert "image" in item["assets"]
    assert "thumb" not in item["assets"]
    assert "meta" not in item["assets"]

def test_cli_recursive_multiple_patterns(test_dir):
    """Test recursive processing with multiple patterns."""
    runner = CliRunner()
    result = runner.invoke(stac, [str(test_dir), "--recursive", "--pattern", "*.tif", "--pattern", "*.json"])
    assert result.exit_code == 0
    item = json.loads(result.output)
    
    assert "image" in item["assets"]
    assert "meta" in item["assets"]
    assert "thumb" not in item["assets"]

def test_cli_directory_without_recursive(test_dir):
    """Test error when providing directory without --recursive."""
    runner = CliRunner()
    result = runner.invoke(stac, [str(test_dir)])
    assert result.exit_code != 0
    assert "is a directory" in result.output
    assert "--recursive" in result.output

def test_cli_recursive_no_files(tmp_path):
    """Test error when directory is empty or no files match."""
    d = tmp_path / "empty"
    d.mkdir()
    runner = CliRunner()
    result = runner.invoke(stac, [str(d), "--recursive"])
    assert result.exit_code != 0
    assert "No valid files found" in result.output
