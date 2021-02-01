from setuptools import setup, find_packages

setup(name = 'meercrab',
      version = '2.0.0',
      description = 'MeerLICHT Classification of Real And Bogus using Deep Learning',
      author = 'Zafiirah Hosenie',
      author_email = 'zafiirah.hosenie@gmail.com',
      license = 'MIT',
      url = 'https://github.com/Zafiirah13/MeerCRAB',
      packages = find_packages(),
      install_requires=['pandas==0.25.3',
                        'tensorflow==1.9',
                        'imbalanced-learn==0.7.0',
                        'matplotlib==3.3',
                        'scipy==1.5.2',
                        'Keras==2.0.9',
                        'Pillow==7.2.0',
                        'scikit_learn==0.23.1',
                        'numpy==1.19.1',
                        'astropy==4.0',
			'h5py'
                        ],
      classifiers=[
                  'Programming Language :: Python :: 3.6.1',
                  ],
      keywords=['Convolutional Neural Network', 'Deep Learning', 'Real and Bogus', 'MeerLICHT'],
      include_package_data = True)

