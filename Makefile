.PHONY:  clean clean-venv venv set-hooks setup build test ruff ruff-check run run-all

IMAGE_REPO=platform
IMAGE_NAME=deeplabcut
LABEL=$(shell cat .ideas/images_spec.json | jq -r ".[0].label")
IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:${LABEL}
LATEST_IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:latest
PLATFORM=linux/amd64
ifndef TARGET
	TARGET=base
endif

# Update the tool specs whenever a new version of a container image is created
TOOL_SPECS=${shell ls -d .ideas/*/tool_spec.json}


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

.DEFAULT_GOAL := build

clean:
	@echo "Cleaning up"
	-docker rmi ${IMAGE_TAG}
	-docker rmi ${IMAGE_TAG}-test

# Builds docker image
# Installs necessary software dependencies for source code
build:
	docker build . -t $(LATEST_IMAGE_TAG) \
		--platform ${PLATFORM} \
		--target ${TARGET}
	docker tag ${LATEST_IMAGE_TAG} ${IMAGE_TAG}
	@$(foreach f, $(TOOL_SPECS), jq --indent 4 '.container_image.label = "${LABEL}"' $(f) > tmp.json && mv tmp.json ${f};)\

# Runs unit tests in docker image
# Used in automated pr checks on github
test: TARGET=test
test: IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:${LABEL}-test
test: LATEST_IMAGE_TAG=${IMAGE_REPO}/${IMAGE_NAME}:latest-test
test: build
	@echo "Running tests..."
	docker run \
		--platform ${PLATFORM} $(RUNTIME) \
		-e USE_GPU=${USE_GPU} \
		--rm \
		${IMAGE_TAG} \
		python -m pytest ${TEST_ARGS}

# Run a tool in the repo
# Specify the tool key to run
run: build
	ideas tools run $(tool) -s -c -n -g all

run-all: build
	@$(foreach f, $(shell ls -d .ideas/*/), ideas tools run -s -c -n -g all $(shell basename $(f));)