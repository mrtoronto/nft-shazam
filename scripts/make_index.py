import logging
from datetime import datetime
import h5py
import re
import pandas as pd
import os
import numpy as np
import json
from scripts.gcs import download_blob, upload_blob
import faiss
import pickle

def make_index(model_type='alexnet', 
				date_suffix=None, 
				collections=None,
				export_prefix='latest'):

	model_dims = {'alexnet': 4096}

	if not date_suffix:
		date_suffix = f"_{datetime.now():%y%m%d%H%M}"

	if not os.path.exists('data'):
		os.mkdir('data')

	if not os.path.exists('data/collections'):
		os.mkdir('data/collections')

	if not os.path.exists('data/archive'):
		os.mkdir('data/archive')

	### Update list of all collections embedded
	if not collections:
		all_cols_filename = f'data/latest_all_collections_embedded_{model_type}.txt'

		download_blob(all_cols_filename, all_cols_filename)

		with open(all_cols_filename, 'r') as f:
			all_cols = f.readlines()

		all_cols = set([c.strip() for c in all_cols if c.strip()])
	else:
		all_cols = set(collections)
	print(f'Making index with collections: {", ".join(all_cols)}')

	metadata = {}
	all_ids = []

	dimension = model_dims[model_type]
	index = faiss.IndexFlatL2(dimension)

	for col in all_cols:
		if not os.path.exists(f'data/collections/{col}'):
			os.mkdir(f'data/collections/{col}')

		col_data_file_path = f"data/collections/{col}/data.json"

		download_blob(col_data_file_path, col_data_file_path, print_info=False)

		with open(col_data_file_path, 'r') as f:
			col_data = json.load(f)

		for v in col_data:
			metadata[f"{col}___{v['token_id']}"] = v

		print(f'Found {len(col_data)} data elements for {col}')

		emb_file_path = f"data/collections/{col}/latest_{model_type}_embeds.h5"
		
		download_blob(emb_file_path, emb_file_path, print_info=False)
		
		with h5py.File(emb_file_path, 'r') as hf:
			col_embs = hf[f"{col}_{model_type}"][:]

		print(f"Found {col_embs.shape[0]} vectors of length {col_embs.shape[1]} in {col}")

		id_filename = f'data/collections/{col}/latest_{model_type}_ids.txt'
		
		download_blob(id_filename, id_filename, print_info=False)
		
		with open(id_filename, 'r') as f:
			col_ids = f.readlines()

		print(f'Found {len(col_ids)} ids for {col}')

		col_ids = [i.strip() for i in col_ids]

		col_embs = col_embs.astype(np.float32)

		index.add(col_embs)

		all_ids += col_ids

	all_ids = {i : v for i, v in enumerate(all_ids)}

	index_file = f"data/{export_prefix}_index_{model_type}.dpr"
	metadata_file = f"data/{export_prefix}_index_metadata_{model_type}.json"
	id_to_meta_mapping = f"data/{export_prefix}_index_id_to_meta_{model_type}.json"

	faiss.write_index(index, index_file)
	with open(metadata_file, mode="w") as f:
		json.dump(metadata, f, indent=4)
	with open(id_to_meta_mapping, mode="w") as f:
		json.dump(all_ids, f, indent=4)

	upload_blob(index_file,index_file)
	upload_blob(metadata_file,metadata_file)
	upload_blob(id_to_meta_mapping,id_to_meta_mapping)

	index_file = f"data/archive/{export_prefix}_index_{model_type}{date_suffix}.dpr"
	metadata_file = f"data/archive/{export_prefix}_index_metadata_{model_type}{date_suffix}.json"
	id_to_meta_mapping = f"data/archive/{export_prefix}_index_id_to_meta_{model_type}{date_suffix}.json"

	faiss.write_index(index, index_file)
	with open(metadata_file, mode="w") as f:
		json.dump(metadata, f, indent=4)
	with open(id_to_meta_mapping, mode="w") as f:
		json.dump(all_ids, f, indent=4)

	upload_blob(index_file,index_file)
	upload_blob(metadata_file,metadata_file)
	upload_blob(id_to_meta_mapping,id_to_meta_mapping)

	if collections:
		all_cols_filename = f'data/{export_prefix}_collections_embedded_{model_type}.txt'

		with open(all_cols_filename, 'w') as f:
			f.write("\n".join(all_cols))

		upload_blob(all_cols_filename, all_cols_filename)

if __name__ == "__main__":
	make_index(export_prefix='punks', collections=['cryptopunks'])