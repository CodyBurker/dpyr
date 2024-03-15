from setuptools import setup
import dpyr
setup(
    name='dpyr',
    # version='0.0.1',
    version = dpyr.__version__,    
    description='A dplyr-like interface for polars.',
    url='https://html-preview.github.io/?url=https://github.com/CodyBurker/dpyr/blob/main/html/dpyr.html',
    author='Cody Burker',
    author_email='codyburker@gmail.com',
    license='BSD 2-clause',
    packages=['dpyr'],
    install_requires=['polars'],
    install_extras=['pandas'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',        
        'Programming Language :: Python :: 3',
    ],
)