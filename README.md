# ideas-toolbox-dlc


**Table of Contents**
- [Toolbox Description](#toolbox-description)
- [How to Get Help](#how-to-get-help)
- [Navigating the Project Repository](#navigating-the-project-repository)


## Toolbox Description
A toolbox for running [DeepLabCut-based](https://github.com/DeepLabCut/DeepLabCut) tools on the IDEAS platform.

This toolbox is designed to run as a Docker image, which can be run on the IDEAS platform. This toolbox consists of the following tools:

- `Train and Evaluate DeepLabCut Model`: Train and evaluate a new DeepLabCut model using a set of labeled images and videos.
- `DeepLabCut Pose Estimation`: Analyze behavioral videos using any pre-trained DeepLabCut model
- `Inscopix Bottom View Mouse Pose Estimation`: Analyze Inscopix bottom-view mouse videos using a pre-trained DeepLabCut model, trained internally at Inscopix.

## How to Get Help
- [IDEAS documentation](https://inscopix.github.io/ideas-docs/tools/dlc/workflow__run_workflow.html) contains detailed information on how to use the toolbox within the IDEAS platform, the parameters that can be used, and the expected output.
- If you have found a bug, we reccomend searching the [issues page](https://github.com/inscopix/ideas-toolbox-deeplabcut/issues) to see if it has already been reported. If not, please open a new issue.
- If you have a feature request, please open a new issue with the label `enhancement`

## Executing the Toolbox

To run the toolbox, you can use the following command:

`make run TOOL=<tool_name>`

Available tools are:

- `inscopix_bottom_up_model__run`
- `training__train_model`
- `workflow__run_workflow`

The command will excute the tool with inputs specified in the `inputs` folder. The output will be saved in the `outputs` folder.

## Navigating the Project Repository

```
├── commands                # Standardized scripts to execute tools on the cloud
├── data                    # Small data files used for testing
├── info                    # Information about the toolbox and its tools
├── inputs                  # Example input files for testing the tools
│── toolbox                 # Contains all code for running and testing the tools
│   ├── tools               # Contains the individual analysis tools
│   ├── utils               # General utilities used by the tools
│   ├── tests               # Unit tests for the individual tools
│── Makefile                 # To automated and standardize toolbox usage
│── Dockerfile               # Commands to assemble the Docker imageons
|── LICENSE                 # License file
|── user_deps.txt           # Python dependencies for the toolbox
└── .gitignore              # Tells Git which files & folders to ignore
```

## License

This project is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en). Note that the software is provided "as is", without warrantee of any kind.