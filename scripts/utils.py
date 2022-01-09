import time, os, json, logging
import faiss
from scripts.gcs import download_blob, upload_blob

def setup_resources(model_type, file_prefix='latest', redownload=False):
	if not os.path.exists(f"/tmp/data"):
		os.mkdir(f"/tmp/data")
	start_time = time.time()

	metadata_filename = f'data/{file_prefix}_index_metadata_{model_type}.json'

	if redownload or not os.path.exists(f"/tmp/{metadata_filename}"):
		download_blob(metadata_filename, f"/tmp/{metadata_filename}")

	with open(f"/tmp/{metadata_filename}", 'r') as f:
		metadata = json.load(f)

	load_time = round(time.time() - start_time, 4)
	logging.info(f'Downloaded metadata in {load_time} s')

	id_to_meta_mapping_filename = f'data/{file_prefix}_index_id_to_meta_{model_type}.json'

	if redownload or not os.path.exists(f"/tmp/{id_to_meta_mapping_filename}"):
		download_blob(id_to_meta_mapping_filename, f"/tmp/{id_to_meta_mapping_filename}")

	with open(f"/tmp/{id_to_meta_mapping_filename}", 'r') as f:
		id_to_meta_mapping = json.load(f)

	load_time = round(time.time() - start_time, 4)
	logging.info(f'Downloaded mapping in {load_time} s')

	os.environ['KMP_DUPLICATE_LIB_OK']='True'

	load_time = round(time.time() - start_time, 4)

	index_filename = f'data/{file_prefix}_index_{model_type}.dpr'

	if redownload or not os.path.exists(f"/tmp/{index_filename}"):
		download_blob(index_filename, f"/tmp/{index_filename}")

	load_time = round(time.time() - start_time, 4)
	logging.info(f'Downloaded index in {load_time} s')

	index = faiss.read_index(f"/tmp/{index_filename}")

	load_time = round(time.time() - start_time, 4)
	logging.info(f'Loaded index in {load_time} s')

	all_cols_filename = f'data/{file_prefix}_collections_embedded_{model_type}.txt'

	download_blob(all_cols_filename, f"/tmp/{all_cols_filename}")

	return metadata, id_to_meta_mapping, index