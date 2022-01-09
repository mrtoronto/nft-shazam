# nft-shazam
Shazam but make it web3

## TL;DR

Use Alexnet to embed images from NFT collections as vectors, use a faiss index to store those embeddings and search them efficiently through an API hostable on a CPU. 

## Overview

Included in this repo is code to,
1. Scrape images from a collection from Opensea
2. Encode the images to vectors using Alexnet
3. Create an index from encoded images
4. Seach the index with a query image

Google Cloud AI Platform was used to run the scraping + encoding job and Google Cloud Storage was used as storage. 

After testing other encoding models (VGG, Densenet and Resnet-18), Alexnet was the best performing of the group.

A FAISS index was used to allow for scaling to hundreds or thousands of collections. The benefits of the FAISS index will become more pronounced once ~100k-1m images are in the database. 

## Running code

### Setup

Setting up this repo requires the basic steps + a few additional requirements. 

#### Basic

First, clone the repo and install the requirements file.
```
git clone https://github.com/mrtoronto/nft-shazam
cd nft-shazam
python3 -m venv venv && source venv/bin/activate && pip3 install -r requirements.txt
```

#### Local settings

After that, you'll need to add a `local_settings.py` file to the `config` folder. 

This file should contain your google cloud service account credential as a dictionary. This credential can be found [here](https://console.cloud.google.com/iam-admin/serviceaccounts) on the service accounts section of the IAM page on Google Cloud Console. 

Select the service account you want and then create a new key. Downland it as a JSON and paste it into your `local_settings.py` file as a variable named `firestore_creds`. 

#### .tar.gz Archive

To run jobs on the AI platform, you'll also need to create a .tar.gz archive of the code for the AI platform to use. This can be done with the setup.py file. 

```
python3 setup.py && gsutil cp dist/nft_shazam-0.1.tar.gz gs://PROJECT_NAME/cloud_function_zips
```

To avoid configuring [gsutil](https://cloud.google.com/storage/docs/gsutil), one can alternatively run `python3 setup.py` and manually upload the .tar.gz file to the proper location on GCS. Make sure the location you choose is properly reflected in the `packageUris` parameter of the `submit_job.py` file. 

### Scraping + Embedding Job

This job takes one or multiple collections, downloads their images from Opensea, encodes each one then uploads the encodings and collection metadata to GCS. Run it for any collections you'd like to add to an index. 

Its run through the AI platform so to submit a job, use the `submit_job.py` script. 

Parameters are edited from within the script. The parameters are,

- model_type
  - `alexnet` is what I used but img_to_vec.py code also supports VGG, densenet and resnet-18
- collections 
  - This should be either a single collection slug or multiple slugs seperated by `, `

Once the parameters are set, run `python3 submit_job.py` and check the AI Platform [Jobs page](https://console.cloud.google.com/ai-platform/jobs) to see if it went through. 

### Building the index

This can be done without a GPU so I ran it locally instead of using the AI platform. 

To run this job, use the following command as a template,
```
python3 -c "from scripts.make_index import make_index; \
make_index(export_prefix='trunc', collections=['pudgypenguins', 'cryptopunks', 'cryptoadz-by-gremplin', 'doodles-official'])"
```

The script will download the embeddings + metadata for each of the collections specified in the `collections` argument and build an index with the data. It will export the files to GCS with the prefix specified with `export_prefix`. 

### Running the API

This can be done by simply running,
```
python3 api.py
```

Once the api is running, you can send a POST request to `/search` with a url in the body. 
```
curl --location --request POST 'http://127.0.0.1:5000/search' \
--form 'url="https://lh3.googleusercontent.com/v2LOStQ2dyt7KX3uj3UxtLzDJJRoVx1XxZesl7bSL-CZxFh3MAEk4FlCj8-suSgT1WtNw9jjwSjXS-olxi0i32tZtNWNqUb2tzbuwTo=w175"' \
--form 'n="1"'
```

returns the following data,
```
[
    {
        "OS_collection": "Cryptoadz",
        "collection_slug": "cryptoadz-by-gremplin",
        "image_url": "https://lh3.googleusercontent.com/v2LOStQ2dyt7KX3uj3UxtLzDJJRoVx1XxZesl7bSL-CZxFh3MAEk4FlCj8-suSgT1WtNw9jjwSjXS-olxi0i32tZtNWNqUb2tzbuwTo",
        "link": "https://opensea.io/assets/0x1cb1a5e65610aeff2551a50f76a87a7d3fb649c6/2667",
        "name": "CrypToadz #2667",
        "score": 380.12353515625,
        "token_id": "2667"
    }
]
```

One can also change to a different set of data by sending a POST request to `/update` with the new prefix in the body.
```
curl --location --request POST 'http://127.0.0.1:5000/update' \
--form 'new_prefix="trunc"'
```

One can also check which collections are included in the currently loaded data by sending a GET request to `/get_collections`. 


## Limitations

- Eventually the index file becomes large which means a server with a lot of RAM will be necessary to serve these requests. When I loaded ~30-40 collections into a single index, the file size was ~4GB which became difficult for my local machine to run without slowing down the rest of the system. This would not be a problem with the right server.
- This method is only applicable to NFTs with images. Videos, audio and gif based NFTs would not be able to use the same architecture without some modifications. 

## Credit

The code used to generate image embeddings is from the img2vec [repo](https://github.com/christiansafka/img2vec) built by christiansafka. Thank you for your help christiansafka. 
