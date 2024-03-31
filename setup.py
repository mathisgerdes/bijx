from setuptools import setup

setup(
    name='jaxcg',
    version='0.1',
    author='Mathis Gerdes',
    author_email='mathisgerdes@gmail.com',
    description='Crouch-Grossmann integration for GL(N) Lie groups.',
    url='https://github.com/lattice-gen/lie-integration',
    packages=['jaxcg'],
    install_requires=[
        'numpy',
        'jax',
        'chex',
    ],
    python_requires='>=3.9',
)