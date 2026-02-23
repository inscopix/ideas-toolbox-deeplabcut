# toolbox variables
REPO=inscopix
PROJECT=ideas
MODULE=toolbox
IMAGE_NAME=dlc
VERSION=$(shell git describe --tags --always --dirty)
IMAGE_TAG=${REPO}/${PROJECT}/${MODULE}/${IMAGE_NAME}:${VERSION}
FULL_NAME=${REPO}/${PROJECT}/${MODULE}/${IMAGE_NAME}
CONTAINER_NAME=${REPO}-${PROJECT}-${MODULE}-${IMAGE_NAME}-${VERSION}
PLATFORM=linux/amd64
PYTHON=python3.10

# This flag determines whether files should be
# dynamically renamed (if possible) after function
# execution.
# You want to leave this to true so that static
# filenames are generated, so that these can be
# annotated by the app.
# If you want to see what happens on IDEAS, you can
# switch this to false
ifndef TC_NO_RENAME
	TC_NO_RENAME="true"
endif

# specify a different data dir to volume mount
# when running the toolbox container for local testing
ifndef DATA_DIR
	DATA_DIR=$(PWD)/data
endif


# vars for gpu functionality
# can set which gpu to use
ifndef GPU_TO_USE
	GPU_TO_USE=0
endif

# if a codebuild id exists, that means the toolbox is being built and tested on aws
# when running unit tests on aws, nvidia runtime is not available
# so we force disable using gpu, so no unit tests which require gpu are executed
ifdef CODEBUILD_BUILD_ID
	USE_GPU=0
endif

ifndef USE_GPU
	USE_GPU=1
endif

# if using gpu, set the runtime to be nvidia
# and configure the gpu
ifeq ($(USE_GPU), 1)
	RUNTIME=--runtime nvidia --gpus all
else
	RUNTIME=
endif

define run_command
    bash -c 'mkdir -p "/ideas/outputs/$1" \
        && cd "/ideas/outputs/$1" \
        && cp "/ideas/inputs/$1.json" "/ideas/outputs/$1/inputs.json" \
        && "/ideas/commands/$1.sh" \
	    && rm "/ideas/outputs/$1/inputs.json"'
endef

.DEFAULT_GOAL := build

clean:
	@echo "Cleaning up"
	-docker rm $(CONTAINER_NAME)
	-docker images | grep $(FULL_NAME) | awk '{print $$1 ":" $$2}' | grep -v $(VERSION) | xargs docker rmi
	-rm -rf $(PWD)/outputs

build:
	@echo "Building docker image..."
	DOCKER_BUILDKIT=1 docker build . -t $(IMAGE_TAG) \
		--platform ${PLATFORM} \
		--target base

test: build clean
	@echo "Running toolbox tests..."
	-mkdir -p $(PWD)/outputs
	docker run \
		--platform ${PLATFORM} $(RUNTIME) \
		-v $(PWD)/data:/ideas/data \
		-v $(PWD)/inputs:/ideas/inputs \
		-v $(PWD)/commands:/ideas/commands \
		-w /ideas \
		-e CODEBUILD_BUILD_ID=${CODEBUILD_BUILD_ID} \
		-e USE_GPU=${USE_GPU} \
		--name $(CONTAINER_NAME) \
		${IMAGE_TAG} \
		$(PYTHON) -m pytest $(TEST_ARGS)


run: build clean
	@bash check_tool.sh $(TOOL)
	@echo "Running the $(TOOL) tool in a Docker container. Outputs will be in /outputs/$(TOOL)"
	docker run \
			--platform ${PLATFORM} $(RUNTIME) \
			-v ${DATA_DIR}:/ideas/data \
			-v $(PWD)/inputs:/ideas/inputs \
			-v $(PWD)/commands:/ideas/commands \
			-e TC_NO_RENAME=$(TC_NO_RENAME) \
			-e USE_GPU=${USE_GPU} \
			-e GPU_TO_USE=${GPU_TO_USE} \
			--name $(CONTAINER_NAME) \
	    $(IMAGE_TAG) \
		$(call run_command,$(TOOL)) \
	&& docker cp $(CONTAINER_NAME):/ideas/outputs $(PWD) \
	&& docker rm $(CONTAINER_NAME)
