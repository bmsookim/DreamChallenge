pwd = ${CURDIR}

preprocessing:
	cd ./dicom-preprocessing && \
	nvidia-docker build \
		-t docker.synapse.org/syn7794493/preprocessing-r2 \
		.
	docker push docker.synapse.org/syn7794493/preprocessing-r2
training:
	cp ./Dockerfiles/training ./train/Dockerfile
	cd ./train && \
	nvidia-docker build \
		-t docker.synapse.org/syn7794493/train-r1 \
		.
	rm ./train/Dockerfile
	docker push docker.synapse.org/syn7794493/train-r1
inference:
	cp ./Dockerfiles/inference ./Dockerfile
	nvidia-docker build \
		-t docker.synapse.org/syn7794493/inference-r1 \
		.
	rm ./Dockerfile
	docker push docker.synapse.org/syn7794493/inference-r1
