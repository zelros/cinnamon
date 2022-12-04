from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cinnamon",
    version="0.1.1",
    author="Yohann Le Faou",
    author_email="lefaou.yohann@gmail.com",
    description="A monitoring tool for machine learning systems that focus on data drift",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zelros/cinnamon",
    project_urls={
        "Source Code": "https://github.com/zelros/cinnamon",
        "Bug Tracker": "https://github.com/zelros/cinnamon/issues",
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    license="MIT",
    python_requires=">=3.6",
    install_requires=["pandas>=1.0",
                      "scikit-learn>=0.24",  # can switch to 1.0 when I switch to Python 3.7 support
                      "xgboost>=1.4",
                      "catboost>=1.0"],
    extras_require={"graphs": ["matplotlib>=3.4.3", "treelib>=1.6.1"]},
    keywords=["data drift", "covariate shift", "concept drift", "monitoring",
              "adversarial learning", "machine learning"],
)
