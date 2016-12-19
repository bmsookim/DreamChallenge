pwd = ${CURDIR}


install:
	cd ./pre-processing && \
	./install.sh
preprocessing:
	cp ./Dockerfile/preprocessing ./pre-processing/Dockerfile
	cd ./pre-processing && \
	nvidia-docker build \
		-t docker.synapse.org/syn7794493/preprocessing-r1 \
		.
	rm ./pre-processing/Dockerfile
training:
	cp ./Dockerfile/training ./train/Dockerfile
	cd ./train && \
	nvidia-docker build \
		-t docker.synapse.org/syn7794493/training-r1 \
		.
	rm ./train/Dockerfile
inference:
	cd ./pre-processing
