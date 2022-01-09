import os
import requests
import json
import logging
from scripts.gcs import download_blob, upload_blob

def get_rel_data(a, collection_name):
	out_data = {
		'image_url': a['image_url'],
		'token_id': a['token_id'],
		'name': a['name'],
		'OS_collection': a['asset_contract']['name'],
		'collection_slug': collection_name,
		'link': a['permalink']
	}
	return out_data


def get_OS_data(limit = 20, 
				collection_name="doodles-official",
				trunc=None):
	offset = 0
	all_assets = []
	params = {'collection': collection_name}
	url = f"https://api.opensea.io/api/v1/assets?"

	if not os.path.exists('data'):
		os.mkdir('data')

	if not os.path.exists('data/collections'):
		os.mkdir('data/collections')

	if not os.path.exists(f"data/collections/{collection_name}"):
		os.mkdir(f"data/collections/{collection_name}")

	while True:
		if offset % 1000 == 0:
			print(f'Found data for {offset} {collection_name} tokens so far')
		loop_url = url
		for i in range(offset, offset + limit):
			loop_url += f'token_ids={i}&'
		
		loop_url = loop_url[:-1]
		response = requests.request("get", loop_url, params=params)
		try:
			assets = json.loads(response.text).get('assets', {})
		except:
			break

		if assets:
			all_assets.extend(assets)

		if trunc and offset >= trunc:
			break
		### continue loop if correct number of assets returned
		elif assets:
			offset = offset + limit
			continue
		else:
			break

	### Remove unnecessary data
	data = [get_rel_data(a, collection_name) for a in all_assets]
	data_filename = f'data/collections/{collection_name}/data.json'
	with open(data_filename, 'w') as f:
		json.dump(data, f, indent=4)

	upload_blob(data_filename, data_filename)
	return data

if __name__ == "__main__":
	all_assets = get_OS_data(collection_name='pudgypenguins')
	all_assets = get_OS_data(collection_name='doodles-official')
