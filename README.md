# ACCESS-Pegasus-Examples

This repository contains preconfigured Pegasus Workflows examples including the Pegasus Tutorial, to run on ACCESS resources.
For first time users, we highly recommend to do the notebooks in order, as they build up on concepts in the previous notebooks.

Table of contents
=================

<!--ts-->
   * [Getting started](#getting-started)
   * [Pegasus Tutorial](#pegasus-tutorial)
      * [Introduction API](#introduction-api)
      * [Debugging](#debugging)
      * [Command Line Tools](#command-Line-Tools)
      * [Summary](#summary)
   * [Workflow Examples](#workflow-examples)
      * [Artificial Intelligence](#artificial-intelligence)
         * [Lung Segmentation](#lung-segmentation)
         * [Mask Detection](#mask-Detection)
         * [Orca Sound](#orca-sound)  
      * [Astronomy](#astronomy)
         * [Montage](#montage) 
      * [Bioinformatics](#bioinformatics)
         * [Rosetta](#rosetta)
         * [Alphafold](#alphafold) 
<!--te-->



Getting started
===============

To get started, use a web browser and log in to https://access.pegasus.isi.edu . Use your ACCESS credentials to log in and start a Jupyter notebook
From there, start a shell by clicking the menu (Clusters > Shell), and check out the repository:

<p align="center">
  <img src="/images/terminal-start.png" style="width: 200px;"/>
</p>

```
$ git clone https://github.com/pegasus-isi/ACCESS-Pegasus-Examples.git
```

In Jupyter, navigate to the example you are interested in, and step through the notebook. Once the workflow is submitted, you have to add compute resources with HTCondor Annex.


Pegasus Tutorial
================

Pegasus tutorial notebook, which is intended for new users who want to get a quick overview of Pegasus concepts and usage. This tutorial covers some of the following 4 chapters:

Introduction API
================

This chapter gives a quick overview of Pegasus and it's Python API, along with the information regarding ACCESS resources supported by it. This chapter covers :
  * Using the Pegasus API to generate an abstract workflow
  * Using the API to plan the abstract workflow into an executable workflow
  * Pegasus catalogs for sites, transformations, and data
 
<p align="center">
  <img src="/images/catalogs.png" style="width: 400px;"/>
</p>


Debugging
=========
When running complex computations (such as workflows) on complex computing infrastructure (for example HPC clusters), things will go wrong. It is therefore important to understand how to detect and debug issues as they appear. The good news is that Pegasus is doing a good job with the detection part, using for example exit codes, and provides tooling to help you debug. This chapter covers some key points involved in the process of debugging, introducing an error and then detecting it. 


Command Line Tools
==================
Running Pegasus in a Jupyter notebook is very convenient for tutorials and for smaller workflows, but production workflows are most commonly submitted on dedicated HTCondor submit nodes using command line tools. This chapter of the tutorial uses command line tools for planning, submitting and checking status of a workflow generated in a notebook. The following command line tools are discussed in the notebook : 
* `pegasus-status -v [wfdir]` Provides status on a workflow instance
* `pegasus-analyzer [wfdir]` Provides debugging clues why a workflow failed. Run this after a workflow has failed
* `pegasus-statistics [wfdir]` Provides statistics, such as walltimes, on a workflow after it has completed
* `pegasus-remove [wfdir]` Removes a workflow from the system


Summary
===============
This chapter covers information regarding how to get in touch with our team, citing Pegasus in your work and contains other useful contact information too.
<p align="center">
  <img src="/images/pegasus-slide.png" style="width: 450px;"/>
</p>


Workflow Examples
=================
Pegasus is currently being used in a broad range of scientific domains. This repository contains the workflow notebooks corresponding to various domains listed here, each notebook shown here is using actual data and executables. This page gives a brief overview of each of the workflow, for more details please go to corresponding folder.


Artificial Intelligence
=======================
Consists of Machine Learning workflows, with examples showing various steps in a conventional machine learning pipeline (data-preprocessing, model training, model evaluation, model inference etc).

Lung Segmentation
=================
We implement a workflow that employs supervised learning techniques to locate lungs on X-ray images. Lung instance segmentation workflow uses Chest X-ray images for predicting lung masks from the images using U-Net model.
<p align="center">
  <img src="/Artificial-Intelligence/LungSegmentation/img/segmentation.png" style="width: 450px;"/>
</p>


Mask Detection
==============
The workflow addresses the problem of determining what percentage of the population is properly wearing masks to better track our collective efforts in preventing the spread of COVID-19 in public spaces. It uses the FastRCNNPredictor deep learning model to detect masks on faces.
<p align="center">
  <img src="/Artificial-Intelligence/MaskDetection/imgs/sample_output.png" style="width: 450px;"/>
</p>


Orca Sound
==========
This workflow processes and analyzes the hydrophone data of sensors from **three locations** in the state of **Washington**, and uses trained machine learning models to automatically identify the whistles of the Orcas.
<p align="center">
  <img src="/Artificial-Intelligence/OrcaSound/images/orca_sound_sensors.png" style="width: 350px;"/>
</p>


Astronomy
=========
The purpose of the examples in this is to illustrate the use of the Pegasus workflows for parallelization of astronomical data processing jobs. 

Montage
=======
This workflow exhibits a standard flow to the processing of a collection of images to make a mosaic. Using Montage processing tools, we create a mosaic of M17 (the Omega Nebula, 1 degree x 1 degree, in the 2MASS J-band). The [Omega Nebula](https://en.wikipedia.org/wiki/Omega_Nebula) (M17) is between 5,000 and 6,000 light-years from Earth and it spans some 15 light-years in diameter.
<p align="center">
  <img src="/Astronomy/Montage/images/M17_location.png" style="width: 250px;"/>
</p>

Bioinformatics
==============

Variant Calling
===============
Pegasus Workflow corresponding to the  [Automating a Variant Calling Workflow](https://datacarpentry.org/wrangling-genomics/05-automation/index.html) 
from Data Carpentry Lesson [Data Wrangling and Processing for Genomics](https://datacarpentry.org/wrangling-genomics/). This workflow downloads and aligns SRA data to the E. coli  REL606 reference genome, and checks what differences exist in our reads versus the genome. The workflow also performs perform  variant calling to see how the population changed over time.

Rosetta
=======
This is a Pegasus workflow for running Rosetta's De novo structure prediction on the OSG. The workflow predicts the 3-dimensional structure of a protein starting with an amino acid sequence, using the [Abinitio Relax](https://new.rosettacommons.org/docs/latest/application_documentation/structure_prediction/abinitio-relax#algorithm) algorithm.

Alphafold
=========
A Pegasus Workflow for running [Alphafold](https://github.com/deepmind/alphafold) model's inference pipeline regarding protein structure
prediction. The current workflow is regarding the Multiple Sequence Alignment (MSA) and 
Feature Generation steps, which produce a `features.pkl` file that can be later used in protein structure inference
stage using the Alphafold model parameters.
