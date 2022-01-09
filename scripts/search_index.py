import logging
import requests
from io import BytesIO
from scripts.img_to_vec import Img2Vec
import faiss
from PIL import Image
import json
import os
from scripts.gcs import download_blob, upload_blob
import time

def search_index(
	metadata, 
	id_to_meta_mapping,
	img2vec,
	index,
	img_path=None, 
	url=None,
	n_returned=3
):
	start_time = time.time()
	assert url or img_path, 'Provide a url or image path'
	if url:
		response = requests.get(url)
		img = Image.open(BytesIO(response.content))
	elif img_path:
		img = Image.open(img_path)

	img = img.convert('RGB')
	### Remove extra channel
	vec = img2vec.get_vec(img, tensor=False)
	vec = vec.reshape(1,-1)

	scores, indexes = index.search(vec, n_returned)

	indexes = indexes[0]
	scores = scores[0]
	out_list = []
	for s, i in zip(scores, indexes):
		meta_id = id_to_meta_mapping[str(i)].strip()
		metadata[meta_id].update({'score': float(s)})
		out_list.append(metadata[meta_id])

	run_time = round(time.time() - start_time, 4)
	print(f'Searched through {index.ntotal} vectors in {run_time}s')

	return out_list

if __name__ == "__main__":
	collection = 'doodles-official'
	token_id = '470'
	file_path = f'/Users/matthewtoronto/Documents/matt/nft_shazam/data/{collection}/images/{token_id}.png'
	file_path = "/Users/matthewtoronto/Downloads/IMG_0808 copy.png"
	file_path = "/Users/matthewtoronto/Downloads/z6vsgusL_400x400.jpg"
	url = "https://pbs.twimg.com/profile_images/1466539219304603648/z6vsgusL_400x400.jpg"
	search_index(url='https://pbs.twimg.com/profile_images/1466539219304603648/z6vsgusL_400x400.jpg')