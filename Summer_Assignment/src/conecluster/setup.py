from setuptools import setup

package_name = 'conecluster'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'scikit-learn'],
    zip_safe=True,
    maintainer='supan',
    maintainer_email='supancshah2005@gmail.com',
    description='Cones clustering node using DBSCAN and marker publishing.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cc = conecluster.ass2:main',
            'try = conecluster.try:main',

           
        ],
    },
)
