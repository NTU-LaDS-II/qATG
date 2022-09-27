from setuptools import setup

setup(
    name='qatg',
    description='An open-source quantum automatic test generator',
    url='https://github.com/MartianSheep/quantum-ATG',
    author='Chin-Yang Jen',
    author_email='b08901132@ntu.edu.tw',
    license='BSD 3-clause',
    packages=['qatg'],
    install_requires=['qiskit>=0.21.2',
					  'pylatexenc>=2.10',
					  'scipy>=1.7.1',
                      'ipympl>=0.9.2',
					  'numpy>=1.20.3'
                      ],

)
