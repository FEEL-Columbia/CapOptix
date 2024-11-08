from setuptools import setup, find_packages

setup(
    name="capoptix",  # Package name
    version="0.1.0",  # Initial version
    author="Millend Roy",
    author_email="mr4404@columbia.edu",
    description="A library for developing capacity premia models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    # url="https://github.com/yourusername/capacity_premia",  # Repository URL
    packages=find_packages(),  # Automatically find subpackages
    install_requires=[
        # List dependencies here, e.g., 'pandas', 'matplotlib'
        "numpy", 
        "croniter",
        "datetime",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "statsmodels",
        "scipy",
        "seaborn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)