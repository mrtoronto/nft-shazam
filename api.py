import time, logging, os, json, faiss
from flask import Flask, request, jsonify
from scripts.search_index import search_index
from scripts.img_to_vec import Img2Vec
from scripts.utils import setup_resources

logging.basicConfig(level=logging.INFO)

model_type = 'alexnet'
file_prefix = 'pengus'

metadata, id_to_meta_mapping, index = setup_resources(
	file_prefix=file_prefix, 
	model_type=model_type
)

app = Flask(__name__)

img2vec = Img2Vec(cuda=False, model=model_type)

@app.route('/search', methods = ['POST'])
def search():
	start_time = time.time()
	if request.method == 'POST':
		url = request.form.get('url')
		n = request.form.get('n', 6)
		results = search_index(
			url=url, 
			metadata=metadata, 
			id_to_meta_mapping=id_to_meta_mapping,
			img2vec=img2vec,
			index=index,
			n_returned=n
		)
		return jsonify(results)

@app.route('/update', methods = ['POST'])
def update():
	if request.method == 'POST':
		global file_prefix
		global metadata
		global id_to_meta_mapping
		global index

		if request.form.get('new_prefix'):
			file_prefix = request.form.get('new_prefix')

		metadata, id_to_meta_mapping, index = setup_resources(
			model_type='alexnet', 
			file_prefix=file_prefix, 
			redownload=True)

		return 'Success'

@app.route('/get_collections', methods= ['GET'])
def get_cols(model_type=model_type):
	global file_prefix
	if request.method == 'GET':
		all_cols_filename = f'data/{file_prefix}_collections_embedded_{model_type}.txt'

		with open(all_cols_filename, 'r') as f:
			all_cols = f.readlines()

		return "".join(all_cols)


if __name__ == '__main__':
    app.run(debug=True)