from datetime import datetime
import h5py
from tqdm import tqdm
import json
import os
import requests
from PIL import Image
from io import BytesIO
from scripts.img_to_vec import Img2Vec
from scripts.gcs import download_blob, upload_blob
import logging
from google.api_core import exceptions
import pandas as pd
import numpy as np


def process_asset(asset, ids, embs, collection_name, img2vec):
	image_folder = f"data/collections/{collection_name}/images"
	url = asset['image_url']
	if not url.strip():
		return ids, embs, False, "No image url"
	filename = f"{image_folder}/{asset['token_id']}.png"

	response = requests.get(url)

	if response.status_code != 200:
		return ids, embs, False, "Status code"

	img = Image.open(BytesIO(response.content))
	img = img.convert('RGB')

	vec = img2vec.get_vec(img, tensor=False)
	vec = vec.tolist()
	embs.append(vec)
	ids.append(f"{collection_name}___{asset['token_id']}")

	return ids, embs, True, ""

def download_OS_data_from_gsc(collection_name, trunc):
	try:
		download_blob(f'data/collections/{collection_name}/data.json',
				f'data/collections/{collection_name}/data.json')
		with open(f'data/collections/{collection_name}/data.json', 'r') as f:
			data = json.load(f)
		logging.info(f'Found {len(data)} tokens in data/{collection_name}/data.json')
		if trunc:
			data = data[:trunc]
		return data
	except exceptions.NotFound as e:
		logging.info(e)
		logging.info('No data found.')
		return None

def export_data(date_suffix, collection_name, model_type, embs, ids):

	emb_filenames = [
		f'data/collections/{collection_name}/latest_{model_type}_embeds.h5', 
		f'data/collections/{collection_name}/{model_type}_embeds{date_suffix}.h5'
	]

	for emb_filename in emb_filenames:
		with h5py.File(emb_filename, 'w') as hf:
			hf.create_dataset(f"{collection_name}_{model_type}",  data=embs)

		upload_blob(emb_filename, emb_filename)

	id_filenames = [
		f'data/collections/{collection_name}/latest_{model_type}_ids.txt', 
		f'data/collections/{collection_name}/{model_type}_ids{date_suffix}.txt'
	]

	for id_filename in id_filenames:
		with open(id_filename, 'w') as f:
			f.write("\n".join(ids))

		upload_blob(id_filename, id_filename)


def update_all_cols_file(model_type, date_suffix, collection_name):

	all_cols_filename = f'data/latest_all_collections_embedded_{model_type}.txt'

	try:
		download_blob(all_cols_filename, all_cols_filename)

		with open(all_cols_filename, 'r') as f:
			all_cols = f.readlines()
	except exceptions.NotFound as e:
		all_cols = []

	all_cols.append(collection_name)

	all_cols = list(set([l.strip() for l in all_cols if l.strip()]))

	### Update list of all collections embedded
	all_cols_filenames = [
		f'data/latest_all_collections_embedded_{model_type}.txt',
		f'data/archive/all_collections_embedded_{model_type}{date_suffix}.txt'
	]
	for all_cols_filename in all_cols_filenames:

		with open(all_cols_filename, 'w') as f:
			f.write("\n".join(all_cols))

		upload_blob(all_cols_filename, all_cols_filename)


def download_images_from_OS(collection_name, 
							trunc=None, 
							model_type='alexnet',
							cuda=False,
							date_suffix=None):
	if not os.path.exists(f"data"):
		os.mkdir(f"data")
	if not os.path.exists(f"data/archive"):
		os.mkdir(f"data/archive")
	if not os.path.exists(f"data/collections"):
		os.mkdir(f"data/collections")
	if not os.path.exists(f"data/collections/{collection_name}"):
		os.mkdir(f"data/collections/{collection_name}")

	if not date_suffix:
		date_suffix = f"_{datetime.now():%y%m%d%H%M}"

	data = download_OS_data_from_gsc(collection_name, trunc)

	if not data:
		return None

	img2vec = Img2Vec(cuda=cuda, model=model_type)

	errors = []
	embs = []
	ids = []
	logging.info(f"Embedding images from {collection_name} with {model_type}")
	for asset_idx, asset in enumerate(data):
		ids, embs, success, error_message = process_asset(asset, ids, embs, collection_name, img2vec)
		if not success:
			logging.warning(f'Failed processing image #{asset_idx} (ID: {asset["token_id"]})')
			logging.warning(f'Error: {error_message}')
			data.pop(asset_idx)
		if asset_idx % 2000 == 0:
			logging.info(f'Embedding asset {asset_idx} / {len(data)}')

	embs = np.vstack(embs)
	logging.info(f"Shape of output array: {embs.shape}")

	export_data(date_suffix, collection_name, model_type, embs, ids)
	
	update_all_cols_file(model_type, date_suffix, collection_name)
	

if __name__ == "__main__":
	download_images_from_OS('pudgypenguins')