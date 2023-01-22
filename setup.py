import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

pth = os.path.dirname(os.path.realpath(__file__))
requirementPath = pth + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setuptools.setup(
    name="GeneticAlgorithmFeaturesSelection",
    version="0.0.3",
    author="Ali Sharifi",
    author_email="alisharifisearch@gmail.com",
    description="Feature Selection with Genetic Algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alisharifi2000/GeneticAlgorithmFeatureSelection",
    project_urls={
        "Bug Tracker": "https://github.com/alisharifi2000/GeneticAlgorithmFeatureSelection/issues",
        "repository": "https://github.com/alisharifi2000/GeneticAlgorithmFeatureSelection"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    install_requires=install_requires
)
