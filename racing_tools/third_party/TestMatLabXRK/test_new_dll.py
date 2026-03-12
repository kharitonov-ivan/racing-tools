
import ctypes
from ctypes import c_int, c_char_p, c_double, POINTER, Structure
import os
import sys

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DLL_PATH = os.path.join(BASE_DIR, "DLL-2022", "MatLabXRK-2022-64-ReleaseU.dll")
XRK_FILE = os.path.join(BASE_DIR, "test.xrk")

print(f"Loading DLL: {DLL_PATH}")
print(f"Loading XRK: {XRK_FILE}")

if not os.path.exists(DLL_PATH):
    print("Error: DLL file not found!")
    sys.exit(1)

if not os.path.exists(XRK_FILE):
    print("Error: XRK file not found!")
    sys.exit(1)

# Mimic TimeStruct from xrk.py / MatLabXRK.h
class TimeStruct(Structure):
    _fields_ = [
        ("tm_sec", c_int),
        ("tm_min", c_int),
        ("tm_hour", c_int),
        ("tm_mday", c_int),
        ("tm_mon", c_int),
        ("tm_year", c_int),
        ("tm_wday", c_int),
        ("tm_yday", c_int),
        ("tm_isdst", c_int),
    ]

try:
    # Attempt to load the DLL
    # On Linux, this requires the file to be a valid ELF shared object (.so)
    # or running under WINE.
    lib = ctypes.cdll.LoadLibrary(DLL_PATH)
except OSError as e:
    print(f"Failed to load DLL: {e}")
    print("Note: On Linux, loading a Windows .dll file directly is not supported.")
    sys.exit(1)

# Define function signatures
lib.get_library_date.restype = c_char_p
lib.get_library_time.restype = c_char_p
lib.open_file.argtypes = [c_char_p]
lib.open_file.restype = c_int
lib.close_file_i.argtypes = [c_int]
lib.close_file_i.restype = c_int
lib.get_vehicle_name.argtypes = [c_int]
lib.get_vehicle_name.restype = c_char_p
lib.get_racer_name.argtypes = [c_int]
lib.get_racer_name.restype = c_char_p

# Test generic info
print(f"Library Date: {lib.get_library_date().decode('utf-8')}")
print(f"Library Time: {lib.get_library_time().decode('utf-8')}")

# Open File
# ctypes requires bytes for char*
file_idx = lib.open_file(XRK_FILE.encode('utf-8'))

if file_idx <= 0:
    print(f"Failed to open file. Error code: {file_idx}")
    sys.exit(1)

print(f"File opened successfully. Index: {file_idx}")

try:
    vehicle = lib.get_vehicle_name(file_idx)
    racer = lib.get_racer_name(file_idx)
    
    print(f"Vehicle: {vehicle.decode('utf-8') if vehicle else 'Unknown'}")
    print(f"Racer: {racer.decode('utf-8') if racer else 'Unknown'}")

finally:
    lib.close_file_i(file_idx)
    print("File closed.")
