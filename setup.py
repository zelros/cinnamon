from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cinnamon",
    version="0.1.0",
    author="Yohann Le Faou",
    author_email="lefaou.yohann@gmail.com",
    description="A monitoring tool for machine learning systems that focus on data drift",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zelros/cinnamon",
    project_urls={
        "Source Code": "https://github.com/zelros/cinnamon",
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BDS License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    license="BSD-3",
    python_requires=">=3.9",
    install_requires=["pandas>=1.2.4", "scipy>=1.7.1", "scikit-learn>=1.0", "xgboost>=1.4.2",
                      "treelib>=1.6.1", "numpy>=1.21.2", "matplotlib>=3.4.3"],
    keywords=["data drift", "covariate shift", "concept drift", "monitoring",
              "adversarial learning", "machine learning"],
)