import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GeneticAlgorithmFeatureSelection",
    version="0.0.2",
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
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9"
)
