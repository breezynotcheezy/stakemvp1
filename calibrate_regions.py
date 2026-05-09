"""
Quick calibration script for screen regions
Run this first to set up the poker table detection zones
"""

from region_calibrator import RegionCalibrator


def main():
    calibrator = RegionCalibrator()
    calibrator.calibrate_all_regions()


if __name__ == "__main__":
    main()
