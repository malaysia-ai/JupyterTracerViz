import setuptools


__packagename__ = 'JupyterTracerViz'

setuptools.setup(
    name=__packagename__,
    packages=setuptools.find_packages(),
    version='0.1',
    python_requires='>=3.8',
    description='Visualize trace.json in Jupyter Notebook cell.',
    author='huseinzol05',
    url='https://github.com/huseinzol05/JupyterTracerViz',
    include_package_data=True
)
