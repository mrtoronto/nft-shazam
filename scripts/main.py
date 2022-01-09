import traceback
from datetime import datetime
from scripts.get_images import download_images_from_OS
from scripts.get_OS_data import get_OS_data
from scripts.make_index import make_index

import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Report statistics')
	parser.add_argument(
		'--task',
		dest='task',
		required=True,
		help='which task to run'
	)
	parser.add_argument(
		'--collection',
		dest='collection'
	)
	parser.add_argument(
		'--collections',
		dest='collections'
	)
	parser.add_argument(
		'--model_type',
		dest='model_type'
	)
	parser.add_argument(
		'--trunc',
		dest='trunc'
	)
	parser.add_argument(
		'--download_images',
		dest='download_images',
		action='store_true'
	)
	parser.add_argument(
		'--save_images',
		dest='save_images',
		action='store_true'
	)
	parser.add_argument(  # useless arg to override an AI platform issue
		'--job-dir',
		help='GCS location to write checkpoints and export models',
		default=''
	)
	args = parser.parse_args()
	return args


def main():
	args = parse_args()

	if args.task == 'download_embed_images':
		date_suffix = f"_{datetime.now():%y%m%d%H%M}"

		collection_name = args.collection
		if args.collections:
			collection_names = [c.strip() for c in args.collections.split(', ') if c.strip()]
		else:
			collection_names = None
		model_type = args.model_type
		trunc = int(args.trunc) if args.trunc else None

		if collection_name:

			get_OS_data(collection_name=collection_name, trunc=trunc)

			download_images_from_OS(
				collection_name=collection_name, 
				model_type=model_type,
				trunc=trunc,
				date_suffix=date_suffix
			)
		elif collection_names:
			for collection_name in collection_names:
				try:
					get_OS_data(collection_name=collection_name, trunc=trunc)

					download_images_from_OS(
						collection_name=collection_name, 
						model_type=model_type,
						trunc=trunc,
						date_suffix=date_suffix
					)

					make_index(model_type=model_type, date_suffix=date_suffix)
				except Exception as e:
					print(traceback.format_exc())
					print(f"Failed on {collection_name}")

	elif args.task == 'make_index':
		date_suffix = f"_{datetime.now():%y%m%d%H%M}"
		model_type = args.model_type
		make_index(model_type=model_type, date_suffix=date_suffix)

	else:
		print('Invalid task')

if __name__ == "__main__":
	main()