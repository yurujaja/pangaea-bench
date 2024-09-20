import os

from setuptools import find_packages, setup

version_py = os.path.join(os.path.dirname(__file__), "geofm_bench", "version.py")
version = open(version_py).read().strip().split("=")[-1].replace('"', "").strip()
read_me = open("README.md", encoding="utf8").read()
setup(
    name="geofm_bench",
    version=version,
    description="Benchmark for Geospatial Foundation Models",
    author="GeoFM Bench Team",
    author_email="",
    url="https://github.com/yurujaja/geofm-bench",
    license="MIT",
    long_description=read_me,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*"]),
)
