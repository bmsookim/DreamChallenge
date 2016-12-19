pwd = ${CURDIR}

preprocessing:
	cp ./Dockerfiles/preprocessing ./pre-processing/Dockerfile
	cd ./pre-processing && \
	nvidia-docker build \
		-t docker.synapse.org/syn7794493/preprocessing-r1 \
		.
	rm ./pre-processing/Dockerfile
training:
	cp ./Dockerfiles/training ./train/Dockerfile
	cd ./train && \
	nvidia-docker build \
		-t docker.synapse.org/syn7794493/training-r1 \
		.
	rm ./train/Dockerfile
inference:
	cp ./Dockerfiles/inference ./Dockerfile
	nvidia-docker build \
		-t docker.synapse.org/syn7794493/inference-r1 \
		.
	rm ./Dockerfile