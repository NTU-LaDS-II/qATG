from setuptools import setup

setup(
    name='qatg',
    description='An open-source quantum automatic test generator',
    url='https://github.com/NTU-LaDS-II/qATG',
    author='Chen-Hung Wu, Chin-Yang Jen, Ching-Huan Wang, Cheng-Yun Hsieh, Jiun-Yun Li, James Chien-Mo Li',
    lab_email='lads427@gmail.com',
    license='BSD 3-clause',
    packages=['qatg'],
    install_requires=['qiskit==1.0.2',
					  'qiskit-aer==0.14.1', #https://github.com/Qiskit/qiskit-aer/issues/2007
					  'pylatexenc>=2.10',
					  'scipy>=1.7.1',
                      'ipympl>=0.9.2',
					  'numpy>=1.20.3'
                      ],

)
