from setuptools import setup

setup(
    name='glutils',
    version='0.1',
    author='Mathis Gerdes',
    author_email='mathisgerdes@gmail.com',
    description='Utility functions for GL(N) Lie groups.',
    url='https://github.com/lattice-gen/lie-integration',
    packages=['glutils'],
    install_requires=[
        'numpy',
        'jax',
        'chex',
        'einops'
    ],
    python_requires='>=3.9',
)