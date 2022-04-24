import os
from setuptools import setup


here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md')) as f:
    readme = f.read()

setup(
    name='torchvision4ad',
    version='0.1.2',
    description='torchvision for anomaly detection',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='A03ki',
    author_email='a03ki04@gmail.com',
    install_requires=['torchvision'],
    url='https://github.com/A03ki/torchvision4ad',
    license='MIT',
    packages=['torchvision4ad', 'torchvision4ad/datasets'],
    keywords='torchvision anomaly detection',
    classifiers=[
        'License :: OSI Approved :: MIT License',
    ]
)
