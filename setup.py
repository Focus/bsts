import setuptools
from pathlib import Path

with open('README.rst', 'r') as fh:
    long_description = fh.read()


def load_req(path):
    reqs = Path(path).read_text().split('\n')
    reqs = [x.strip() for x in reqs]
    return [x for x in reqs if len(x) > 0]


setuptools.setup(
    name='bsts',
    version='0.1',
    author='Bati Sengul',
    description='Python library for Bayesian structural time series',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(
        include=['bsts', 'bsts.*'],
    ),
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.6',
    install_requires=load_req('requirements.in'),
    include_package_data=True,
)
