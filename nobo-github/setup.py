from setuptools import setup

setup(
    name="nobo",
    version="1.0",
    description="The module that the guy from the repo should have created",
    author="thomasahle",
    author_email="thomas@ahle.dk",
    packages=["nobo"],  # same as name
    install_requires=[
        "botorch",
        "scikit-optimize",
        "gpytorch",
        "numpy",
        "torch",
        "scipy",
    ],  # external packages as dependencies
)
