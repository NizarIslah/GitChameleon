DOCKER_IMAGE := gc
DOCKER_TAG := 1.0
PYTHON_VERSION ?= 3.9

.PHONY: docker-run

docker-build:
	docker build . -t $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-run:
	docker run --rm -it -v ".:/app/repo" $(DOCKER_IMAGE):$(DOCKER_TAG) -c "pyenv global $(PYTHON_VERSION) && exec bash"
