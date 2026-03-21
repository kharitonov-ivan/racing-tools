import sys
from pathlib import Path
import pytest
import pandas as pd

# Add converter package to path
# Assuming structure:
# racing-tools/
#   converter/
#     converter/ (source)
#     tests/
top_level = Path(__file__).resolve().parents[2]
converter_dir = top_level / "converter"
if str(converter_dir) not in sys.path:
    # Append the root 'converter' folder so imports like 'from converter.session' might work
    # But wait, the Session class is in converter/converter/session.py
    # If the package is 'converter', we should import from expected namespace.
    # The existing code did sys.path.append(str(converter_src / "converter")) -> from session import ...
    # We will try to mimic that to avoid import errors
    sys.path.append(str(converter_dir / "converter"))

try:
    from session import Session
except ImportError:
    # If not found, maybe retry appending parent
    sys.path.append(str(converter_dir))
    from converter.session import Session


@pytest.fixture
def data_dir():
    return Path(__file__).parent / "data"

def test_load_aim_csv(data_dir):
    path = data_dir / "aim_csv" / "aim_csv.csv"
    if not path.exists():
        pytest.skip("AIM CSV data missing")
        
    session = Session.load_aim_csv(path)
    assert isinstance(session, Session)
    assert not session.table.empty
    assert "Time" in session.table.columns
    assert "GPS Speed" in session.table.columns
    
    print(f"AIM CSV loaded: {len(session.table)} rows")
    print(repr(session))
    print("-" * 50)

def test_load_alfano_csv(data_dir):
    folder = data_dir / "alfano_csv" 
    if not (folder / "Excel_SN8448_301025_17H02_Nikoloz_P_RUSTAVI_INTERNATIONAL_MOTORPARK.csv").exists():
        pytest.skip("Alfano Excel CSV data missing")
        
    session = Session.load_alfano_csv(folder)
    assert isinstance(session, Session)
    assert not session.table.empty
    assert "Time" in session.table.columns
    
    print(f"Alfano Excel loaded: {len(session.table)} rows")
    print(repr(session))
    print("-" * 50)

def test_load_gpx(data_dir):
    path = data_dir / "gpx" / "GX010317_1_GPS9.gpx"
    if not path.exists():
        pytest.skip("GPX data missing")
    
    try:
        import gpxpy
    except ImportError:
        pytest.skip("gpxpy not installed")

    session = Session.load_gpx(path)
    assert isinstance(session, Session)
    assert not session.table.empty
    assert "Time" in session.table.columns
    assert "GPS Latitude" in session.table.columns
    assert "GPS Longitude" in session.table.columns
    
    # Check computed speed
    assert "GPS Speed" in session.table.columns 
    
    # Check Distance
    assert "Distance" in session.table.columns
    
    # Check Altitude (renamed from GPS Altitude by normalizer)
    assert "Altitude" in session.table.columns or "GPS Altitude" in session.table.columns
    
    print(f"GPX loaded: {len(session.table)} rows")
    print(repr(session))
    print("-" * 50)

def test_load_aim_raw(data_dir):
    path = data_dir / "aim_raw" / "KharitonovIvan_RotaxMax_ABCD_a_3781.xrk"
    if not path.exists():
        pytest.skip("AIM RAW data missing")
        
    try:
        session = Session.load_aim_raw(path)
    except (ImportError, OSError) as e:
        pytest.skip(f"XRK library or DLL not working: {e}")
    except Exception as e:
        pytest.fail(f"load_aim_raw failed: {e}")

    assert isinstance(session, Session)
    assert not session.table.empty
    print(f"AIM Raw loaded: {len(session.table)} rows")
    print(repr(session))
    print("-" * 50)

def test_load_dispatcher(data_dir):
    # Test GPX dispatch
    gpx_path = data_dir / "gpx" / "GX010317_1_GPS9.gpx"
    if gpx_path.exists():
        try:
             import gpxpy
             s = Session.load(gpx_path)
             assert isinstance(s, Session)
             assert s.device == "GPX"
        except ImportError:
             pass

    # Test AIM Raw dispatch
    xrk_path = data_dir / "aim_raw" / "KharitonovIvan_RotaxMax_ABCD_a_3781.xrk"
    if xrk_path.exists():
        try:
            s = Session.load(xrk_path)
            assert isinstance(s, Session)
            assert s.device == "AIM XRK"
        except (ImportError, OSError):
            pass # Skip if DLL/TDA fails, but dispatch logic ran

    # Test Alfano CSV dispatch (folder)
    alfano_folder = data_dir / "alfano_csv"
    if (alfano_folder / "Excel_SN8448_301025_17H02_Nikoloz_P_RUSTAVI_INTERNATIONAL_MOTORPARK.csv").exists():
        s = Session.load(alfano_folder)
        assert isinstance(s, Session)
        # Alfano loader doesn't explicitly set 'device' in current code snippet? 
        # checking load_alfano_csv source would confirm. 
        # But at least it returns a Session.

    # Test AIM CSV dispatch (file)
    aim_csv_path = data_dir / "aim_csv" / "aim_csv.csv"
    if aim_csv_path.exists():
        s = Session.load(aim_csv_path)
        assert isinstance(s, Session)

    # Test unsupported
    # create a dummy file to pass is_file() check
    dummy_path = data_dir / "dummy.txt"
    dummy_path.touch()
    try:
        with pytest.raises(ValueError, match="Unsupported file extension"):
            Session.load(dummy_path)
    finally:
        dummy_path.unlink()

