import setuptools


setuptools.setup(
    name='scAMACE_py',
    version='1.0',
    packages=setuptools.find_packages(where="scAMACE_py"),
    url='https://github.com/WWJiaxuan/scAMACE_py.git',
    # license='',
    install_requires=['numpy','torch','scipy'],
    author='Zexuan SUN, Jiaxuan WANGWU',
    author_email='wwjiaxuan@link.cuhk.edu.hk',
    description='scAMACE(integrative Analysis of single-cell Methylation, chromatin ACcessibility, and gene Expression) python implementation, a model-based approach to the joint analysis of single-cell data on chromatin accessibility, gene expression and methylation.',
    include_package_data=True,
    package_data={'': ['data/*.csv']}
)
