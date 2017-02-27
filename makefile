pwd = ${CURDIR}

preprocessing-r2:
	cd ./dicom-preprocessing && \
	nvidia-docker build \
		-t docker.synapse.org/syn7794493/preprocessing-r2 \
		.
	docker push docker.synapse.org/syn7794493/preprocessing-r2
preprocessing:
	cd ./dicom-preprocessing && \
	nvidia-docker build \
		-t docker.synapse.org/syn7794493/preprocessing-r3 \
		.
	docker push docker.synapse.org/syn7794493/preprocessing-r3
training:
	cp ./Dockerfiles/training ./train/Dockerfile
	cd ./train && \
	nvidia-docker build \
		-t docker.synapse.org/syn7794493/train-r3 \
		.
	rm ./train/Dockerfile
	docker push docker.synapse.org/syn7794493/train-r3
inference:
	cp ./Dockerfiles/inference ./Dockerfile
	nvidia-docker build \
		-t docker.synapse.org/syn7794493/inference-r1 \
		.
	rm ./Dockerfile
	docker push docker.synapse.org/syn7794493/inference-r1
