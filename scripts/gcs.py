import pathlib
import shutil
import os
import json
from config.local_settings import firestore_creds
import logging
from google.cloud import storage

def upload_blob(source_file_name, 
				dest_filename, 
				folder_name = None, 
				bucket_name='nft-shazam',
				print_info=True):
	"""Uploads a file to the bucket."""
	
	creds_filename = '/tmp/creds.json'
	with open('/tmp/creds.json', 'w') as f:
		json.dump(firestore_creds, f)
	storage_client = storage.Client.from_service_account_json(creds_filename)

	bucket = storage_client.get_bucket(bucket_name)
	### if source_file_name is single file
	if os.path.isfile(source_file_name):

		blob = bucket.blob(dest_filename)

		blob.upload_from_filename(source_file_name)
		if print_info:
			logging.info(f"File {source_file_name} uploaded to {dest_filename}.")
	### if source_file_name is a directory
	elif os.path.isdir(source_file_name):
		### Upload each file in the directory
		for file in os.listdir(source_file_name):
			blob_location = f'{dest_filename}/{file}'
			blob = bucket.blob(blob_location)
			blob.upload_from_filename(f'{source_file_name}/{file}')
			if print_info:
				logging.info(f"File {source_file_name}/{file} uploaded to {blob_location}.")


def download_blob(source_blob_name, 
					destination_file_name, 
					bucket_name='nft-shazam', 
					run_locally=False, 
					folder=False,
					print_info=True):
	"""Downloads a blob from the bucket."""

	creds_filename = '/tmp/creds.json'
	with open('/tmp/creds.json', 'w') as f:
		json.dump(firestore_creds, f)
	storage_client = storage.Client.from_service_account_json(creds_filename)

	bucket = storage_client.bucket(bucket_name)
	
	if not folder:
		blob = bucket.blob(source_blob_name)
		blob.download_to_filename(destination_file_name)
		if print_info:
			logging.info(f"Blob {blob.name} downloaded to {destination_file_name}.")
	else:
		if print_info:
			logging.info(f'{source_blob_name} is a directory. Downloading files individually.')
		shutil.rmtree(destination_file_name)
		pathlib.Path(destination_file_name).mkdir(parents=True, exist_ok=True)

		for blob in bucket.list_blobs(prefix=source_blob_name):
			blob.name = blob.name.split('/')[-1]
			if blob.name:
				blob.download_to_filename(f'{destination_file_name}/{blob.name}')
				if print_info:
					logging.info(f"Blob {blob.name} downloaded to {destination_file_name}/{blob.name}.")
