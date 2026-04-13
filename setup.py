from setuptools import setup, find_packages

setup(
    name='qatg',
    description='An open-source quantum automatic test generator with online test and error mitigation (OTEM), compatible with Qiskit.',
    url='https://github.com/NTU-LaDS-II/qATG',
    author='Chen-Hung Wu, Chin-Yang Jen, Ching-Huan Wang, Cheng-Yun Hsieh, Cynthia Kuan, Hsien-Fu Hsiao, John Wei, Jiun-Yun Li, James Chien-Mo Li',
    author_email='lads427@gmail.com',
    maintainer='NTU-LADS-II',
    license='BSD 3-clause',
    packages=find_packages(exclude=['experiment', 'experiment.*', 'examples', 'docker']),
    python_requires='>=3.9',
    install_requires=[
        'qiskit>=1.0.2',
        'qiskit-aer>=0.14.1',
        'pylatexenc>=2.10',
        'scipy>=1.7.1',
        'ipympl>=0.9.2',
        'numpy>=1.20.3',
    ],
    extras_require={
        'otem': [
            'statsmodels>=0.13',
            'qiskit-ibm-runtime>=0.20',
            'matplotlib>=3.5',
        ],
    },
)
