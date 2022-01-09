import datetime
import logging
from googleapiclient import discovery
from googleapiclient.errors import HttpError
from google.oauth2 import service_account

def submit_job(task_name, scale_tier=None, extra_args=None, master_type=None, accelerator_type=None):

	job_name = task_name + "_" + datetime.datetime.now().strftime("%y%m%d_%H%M%S")

	project_id = f"projects/matts-nlp-jobs"
	args = ['--task', task_name]

	if extra_args:
		args += extra_args

	if not scale_tier:
		scale_tier = 'BASIC'

	training_inputs = {
		'args': args,
		'runtimeVersion': '2.1',
		'pythonVersion': '3.7',
		'scaleTier': scale_tier,
		'packageUris': ["gs://nft-shazam/cloud_function_zips/nft_shazam-0.1.tar.gz"],
		'pythonModule': 'scripts.main',
		'region': 'us-central1',
		'jobDir': "gs://nft-shazam",
	}

	if scale_tier == 'CUSTOM':
		training_inputs.update({'masterType': master_type})

	if accelerator_type:
		training_inputs.update({'masterConfig': {'acceleratorConfig': {'count': 1, 'type': accelerator_type}}})

	scopes = ['https://www.googleapis.com/auth/cloud-platform']
	credentials = service_account.Credentials.from_service_account_file('config/sa.json', scopes=scopes)
	job_spec = {"jobId": job_name, "trainingInput": training_inputs}
	cloudml = discovery.build("ml", "v1", cache_discovery=False, credentials=credentials)
	request = cloudml.projects().jobs().create(body=job_spec, parent=project_id)

	try:
		request.execute()
	except HttpError as err:
		logging.error(
			f'There was an error creating the training job. '
			f'Check the details: {err._get_reason()}'
		)

def download_embed_images_Task(event="", context=""):
	extra_args = ['--model_type', 'alexnet',
				'--collections', 'non-fungible-olive-gardens, lilpudgys, uncool-cats-nft, creaturetoadz, cryptoflyz, niftydegen, eponym, obitsofficial, pussyriotacab, supermetalmons, zombietoadzofficial, lockdown-lemmings'
				# '--collections', 'adam-bomb-squad'
				]
	
	submit_job(task_name='download_embed_images', 
				scale_tier='BASIC_GPU',
				extra_args=extra_args)

def remake_index_Task(event="", context=""):
	extra_args = ['--model_type', 'alexnet']
	submit_job(task_name='make_index', 
				scale_tier='STANDARD_1',
				extra_args=extra_args)