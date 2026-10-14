# Longitudinal tracking of multiple sclerosis lesions in the spinal cord: A validation study

TODO: if accepted add icon here

<img src="https://github.com/user-attachments/assets/cdb12ab9-aa16-42b9-b0f2-8fcfbf2e8062" width="900"/>

Official repository of the longitudinal lesion tracking of MS SC lesions.

This repo contains the code for evaluating 5 strategies to track SC MS lesions across timepoints. More details can be found in the paper.

```
TODO: if accepted add the citation here
```

## Tracking strategies

Some scripts refer to the strategies by their number (e.g. `strat5`).

| # | Strategy | Script |
|---|---|---|
| 1 | Hungarian algorithm in the SC coordinate system (unregistered) | [track_lesion_unreg.py](./tracking/track_lesion_unreg.py) |
| 2 | XGBoost using lesion features in the SC coordinate system | [track_lesion_xgb.py](./tracking/track_lesion_xgb.py) |
| 3 | Siamese network using lesion features in the SC coordinate system | [track_lesion_siamese.py](./tracking/track_lesion_siamese.py) |
| 4 | Hungarian matching of lesion centroids in the baseline scan reference space (registered + CoM) | [track_lesion_reg_com.py](./tracking/track_lesion_reg_com.py) |
| 5 | Lesion overlap in the baseline scan reference space (registered + IoU) | [track_lesion_reg_iou.py](./tracking/track_lesion_reg_iou.py) |

## Installation

1. Install [SCT v7.4](https://spinalcordtoolbox.com/user_section/installation.html)

2. Clone the repository and install the Python dependencies (Python 3.10):

```console
git clone https://github.com/ivadomed/sc-ms-lesion-tracking.git
cd sc-ms-lesion-tracking
conda create -n venv_tracking python=3.10 -y
conda activate venv_tracking
pip install -r requirements.txt
```

## Repository structure

<details>
<summary><b>Repository structure</b></summary>

| Folder | Content |
|---|---|
| [data_preparation](./data_preparation/) | Build the MSD datalist, label the GT lesion segmentations and create the lesion mapping files, anonymize the derivatives |
| [lesion_segmentation](./lesion_segmentation/) | Run the automatic lesion segmentation on CanProCo |
| [tracking](./tracking/) | Lesion tracking methods (unregistered, registered + CoM, registered + IoU, XGBoost, Siamese), SCT segmentation/registration and building of the csv file for ML methods |
| [cross_validation](./cross_validation/) | Leave-one-subject-out cross-validation of the unregistered, XGBoost and Siamese methods |
| [hyperparameter_sweeps](./hyperparameter_sweeps/) | Sweeps of `w_z_over_disk` (unregistered) and of the IoU threshold (registered + IoU) |
| [evaluation](./evaluation/) | Evaluation of the tracking methods and statistical comparison |
| [registration_benchmark](./registration_benchmark/) | Comparison of SCT registration methods |
| [visualization](./visualization/) | Maximum intensity projections of the label images |
| [utils.py](./utils.py) | Shared functions (SC and lesion segmentation, centerline, vertebral levels, lesion labeling ...) |

</details>


## Reproducing the study 

<details>
<summary><b>Here are the steps to reproduce the study:</b></summary>

### Data preparation

1. Build the MSD file from the include.yml file of selected subjects for this study (in this case we used this [include.yml](./data_preparation/include.yml) file)

```console
python build_msd_from_include_yml.py --include-yml /path/to/include.yml --dataset-path /path/to/dataset --output-path /path/to/output/folder
```

2. Generate the GT lesion mappings for GT lesion segmentations
        
    To do so, I labeled the lesion segmentations (so that each connected-component has a unique instance value) using  [generate_gt_labeled_segs_with_mapping.py](./data_preparation/generate_gt_labeled_segs_with_mapping.py). Then I manually filled the created lesion_mapping.json file without considering how they were initially filled.

    NB: for the creation of the lesion mapping for predicted lesion segmentations, I just use the initialized file from strategy #5 (reg+IoU) and correct the mappings manually.

3. Generate the segmentation files using SCT with [run_sct_methods.py](./tracking/run_sct_methods.py)

```console
python run_sct_methods.py -i /path/to/msd/dataset -o /output/path
```

4. Generate the csv file for ML lesion tracking with [generate_csv_for_ML.py](./tracking/generate_csv_for_ML.py)

```console
python generate_csv_for_ML.py  -i /path/to/msd/ -pred /path/to/pred/segs/ -gt-mapping /path/to/folder/GT/mappings -o /path/output/folder 
```

### Running lesion tracking

5. Run tracking for non-ML methods using [run_lesion_tracking.py](./tracking/run_lesion_tracking.py)

```console
 python run_lesion_tracking.py -i /path/to/msd -pred /path/to/pred/segs -o /path/output/folder -m registered_with_CoM OR registered_with_IoU OR unregistered
```

6. Run tracking with XGB using [track_lesion_xgb.py](./tracking/track_lesion_xgb.py)

```console
python track_lesion_xgb.py --data /path/to/csv/dataset --output-folder /path/output/folder
```

7. Run tracking with a Siamese model using [track_lesion_siamese.py](./tracking/track_lesion_siamese.py)

```console
python track_lesion_siamese.py --data /path/to/csv/dataset --output-folder /path/output/folder
```

### Lesion tracking evaluation

8. To evaluate each tracking method we run [run_evaluation.py](./evaluation/run_evaluation.py)


</details>
