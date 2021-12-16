from setuptools import setup, find_packages

with open("napari_text_layer/__init__.py", encoding="utf-8") as f:
    line = next(iter(f))
    VERSION = line.strip().split()[-1][1:-1]
      
with open("README.md", "r") as f:
    readme = f.read()
    
setup(
    name="napari-magic-glasses",
    version=VERSION,
    description="Real time local image analysis in napari.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Hanjin Liu",
    author_email="liuhanjin-sc@g.ecc.u-tokyo.ac.jp",
    license="BSD 3-Clause",
    download_url="https://github.com/hanjinliu/napari-magic-glasses",
    entry_points={"napari.plugin": "napari-magic-glasses = napari_magic_glasses"},
    packages=find_packages(),
    classifiers=["Framework :: napari",
                 "Programming Language :: Python",
                 "Programming Language :: Python :: 3",
                 "Programming Language :: Python :: 3.8",
                 "Programming Language :: Python :: 3.9",
                 ],
    python_requires=">=3.8",
    )