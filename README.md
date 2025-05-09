# Tuberculolis LAM Test Diagnosis


## Introduction 

This project uses computer vision and machine learning to automate the classification of TB LAM (Lipoarabinomannan) diagnostic test results. It aims to improve tuberculosis detection accuracy by analyzing test images, particularly for use in low-resource healthcare settings.


This is a research project that attempts to use machine learning to classify whether an HIV patient has tuberculosis or not based on a LAM test image. This was done in collaboration with a team from the PHRU in South Africa.

---

## Overview

Tuberculosis diagnonisis in the PHRU is currently a manual and laborious process. Nurses will take a photo of a LAM test and send it to a doctor via direct message and receive a diagnosis. This is time consuming, and we attempt to automate this process using machine learning. Our approach is to do the following:

- Use a pre-trained open source model for COVID-19 diagonis as motivation and baseline for our model
- Preprocess images to ensure they can be standardized for the model
- Further fine-tune the COVID-19 model for our specific case 
- Evaluate the model on a key set of performance metrics and statistical significance tests

---

## File Structure

Here is the file structure of the project:

`original_model_train.py`: script to train the original open-source COVID-19 model

`tblam_d.py`, `tblam_finetune.py`: scripts to finetune the generated COVID-19 models for our specific use case

`generatesynthetic.py`: script meant to combat data imbalances through creating synthetic data

`evaluate_synthetic.py`: script that evaluates synthetic data on key use cases

`run.py`: script to generate more performance metrics on the original open-source model

`preprocessing_images/preprocess_images.py`: script to preporcess PHRU images for model

`preprocessing_images/organize_images.py`: script to organize the PHRU files from sharepoint into a folder

`requirements.txt`: list of Python packages needed for the project 

`Model_D_results.csv`: output evaluation for model D specifically for debugging

`model_comparison_results.csv`: model comparison results for the models 

## Running the project:

In order to run the project, first ensure you have the open source model installed. You can do this by running the following command:

```
python original_model_train.py
```

This will train the open-source COVID-19 model locally and save the models locally.

Make sure you then have the PHRU images from the sharepoint folder locally as the `Dataset` folder.

After this, run the preprocessing script in the `preprocessing_images` folder to standardize the images for the model.

```
python preprocess_images.py
```

Once preporcessed, you can run the `tblam_d.py`, `tblam_finetune.py` scripts to fine-tune the open source model for our use case. Please note we have two different scripts because model D was giving some errors when we tried to finetune it the first time. 

Those are the main steps to replicate this project. You will see key data in the `model_comparison_results.csv` file as well as graphs in these folders:

`graphs/`, `comparison_plots/`, `saliency_maps/`, `difficulty_analysis/`, `training_plots/`

If you want to further optimize performance, you can create the synthetic data by running the `generatesynthetic.py` script.

```
python generatesynthetic.py
```

This will place all the newly created synthetic data into the `Dataset` folder. You can then run the `evaluate_synthetic.py` script to generate critical evaluation metrics.


## Conclusion

This project mangified exisiting data imbalance issues in the original PHRU dataset. We had 209 indeterminate, 420 negative, and 76 positive test cases, making it very difficult to train a model with high accuracy, even with all the opitmizations with fine tuning and synthetic data generation. Further work should improve this imbalance but can use this project as a baseline to run models once this imbalance is addressed.