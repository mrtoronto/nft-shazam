from setuptools import find_packages
from setuptools import setup

setup(
	name='nft_shazam',
	version='0.1',
	packages=find_packages(),
	install_requires=["faiss-gpu", 
					"torch",
					"torchvision",
					"tqdm", 
					"pillow",
					"h5py"],
	include_package_data=True,
	description='Gen poetry'
)