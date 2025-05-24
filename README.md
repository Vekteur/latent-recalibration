# Latent Recalibration

This is the repository associated with the paper [Multivariate Latent Recalibration for Conditional Normalizing Flows](https://arxiv.org/abs/2505.16636).

<p align="center">
<img src="images/visualization.png?raw=true" alt="" width="95%" align="top">
</p>

**Abstract**: Reliably characterizing the full conditional distribution of a multivariate response variable given a set of covariates is crucial for trustworthy decision-making. However, misspecified or miscalibrated multivariate models may yield a poor approximation of the joint distribution of the response variables, leading to unreliable predictions and suboptimal decisions. Furthermore, standard recalibration methods are primarily limited to univariate settings, while conformal prediction techniques, despite generating multivariate prediction regions with coverage guarantees, do not provide a full probability density function. We address this gap by first introducing a novel notion of latent calibration, which assesses probabilistic calibration in the latent space of a conditional normalizing flow. Second, we propose latent recalibration (LR), a novel post-hoc model recalibration method that learns a transformation of the latent space with finite-sample bounds on latent calibration. Unlike existing methods, LR produces a recalibrated distribution with an explicit multivariate density function while remaining computationally efficient. Extensive experiments on both tabular and image datasets show that LR consistently improves latent calibration error and the negative log-likelihood of the recalibrated models.

## Datasets

For convenience, all datasets except MEPS and AFHQ are directly provided in this repository. MEPS requires accepting data usage agreements [[1]](https://meps.ahrq.gov/data_stats/download_data/pufs/h181/h181doc.shtml#Data) and [[2]](https://meps.ahrq.gov/data_stats/download_data/pufs/h192/h192doc.shtml#DataA) (see step 3 of the installation). Licenses are detailed [below](#dataset-licenses).

## Example usage

The following code shows an example usage of the code in this repository.

```python
import torch

from moc.configs.config import get_config
from moc.recalibration import LatentRecalibrator
from moc.datamodules.real_datamodule import RealDataModule
from moc.metrics.distribution_metrics import nll
from moc.metrics.calibration import latent_calibration_error
from moc.models.mqf2.lightning_module import MQF2LightningModule
from moc.models.trainers.lightning_trainer import get_lightning_trainer
from moc.utils.run_config import RunConfig

# Data loading and model training
config = get_config()
config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
rc = RunConfig(config, 'mulan', 'sf2')
datamodule = RealDataModule(rc)
p, q = datamodule.input_dim, datamodule.output_dim
model = MQF2LightningModule(p, q)
trainer = get_lightning_trainer(rc)
trainer.fit(model, datamodule)
model.eval()

# Recalibration
recalibrated_model = LatentRecalibrator(model, datamodule)

# Evaluation on one test batch
test_batch = next(iter(datamodule.test_dataloader()))
x, y = test_batch
x, y = x.to(model.device), y.to(model.device)
dist = recalibrated_model.predict(x)
with torch.no_grad():
    nll_value = nll(dist, y).mean()
    latent_calibration = latent_calibration_error(dist, y)
print(nll_value)
print(latent_calibration)
```

## Installation

### Prerequisites
- Python >= 3.9

### Steps
1. (Optional) Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install the package with its dependencies:
```bash
pip install .
```
For exact versions ensuring reproducibility, use instead:
```
pip install -r requirements.txt
```

3. (Optional) For running experiments on MEPS datasets, read the data usage agreements and download the datasets according to [these instructions](https://github.com/yromano/cqr/tree/master/get_meps_data), summarized below:
```bash
git clone https://github.com/yromano/cqr
cd cqr/get_meps_data/
Rscript download_data.R
python main_clean_and_save_to_csv.py
cd ../../
for id in 19 20 21; do mv "cqr/get_meps_data/meps_${id}_reg.csv" "data/feldman/meps_${id}.csv"; done
rm -rf cqr
```

4. (Optional) For running experiments on AFHQ, use [instructions from clovaai/stargan-v2](https://github.com/clovaai/stargan-v2/blob/master/download.sh) summarized below:
```bash
URL=https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0
ZIP_FILE=./data/afhq.zip
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./data
rm $ZIP_FILE
```

## Reproducing the results

To compute the main results of the paper:
```
python run.py name="lr" datasets="real" repeat_tuning=10 tuning_type="lr_mqf2" device="cuda" default_batch_size=1024
```
Other results can be computed using:
```
# Masked auto-regressive flow
python run.py name="lr" datasets="real" repeat_tuning=10 tuning_type="lr_arflow" device="cuda" default_batch_size=512
# Misspecified model
python run.py name="lr" datasets="real" repeat_tuning=10 tuning_type="lr_mqf2" device="cuda" default_batch_size=1024
# TarFlow with noisy targets
python run.py name="lr_tarflow_noisy" datasets="afhq" repeat_tuning=20 tuning_type="lr_tarflow" default_batch_size=256 device="cuda" only_cheap_metrics=True afhq_noise=0.07
# TarFlow without noise on the targets
python run.py name="lr_tarflow_no_noise" datasets="afhq" tuning_type="lr_tarflow" default_batch_size=256 device="cuda" only_cheap_metrics=True afhq_noise=0
```

Then, plots and tables can be generated using `analysis_lr.ipynb`.

Other figures can be generated using `visualizations_lr.ipynb`.

## Dataset licenses

This project utilizes data from various sources, each with its own licensing terms. The table below details the origin and license for each dataset.

| Dataset Group / Name             | Data Source(s)                                                                                                                                                                                                                                             | License(s)                                                                                                                                                                                                      |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Mulan Datasets**               | Main repository: [tsoumakas/mulan](https://github.com/tsoumakas/mulan) <br/> Original sources: [README](https://github.com/tsoumakas/mulan/blob/master/data/multi-target/README.md)                                                                        | GPL                                                                                                                                                                                                             |
| **house, bio, blog\_data**       | [bio](https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure), [house](https://www.kaggle.com/c/house-prices-advanced-regression-techniques), [blog\_data](https://archive.ics.uci.edu/ml/datasets/BlogFeedback) | CC BY 4.0                                                                                                                                                                                                       |
| **meps\_19, meps\_20, meps\_21** | Data from the [Agency for Healthcare Research and Quality (MEPS)](https://meps.ahrq.gov/mepsweb/)                                                                                                                                                          | Requires accepting Data Usage Agreements [1](https://meps.ahrq.gov/data_stats/download_data/pufs/h181/h181doc.shtml#Data) and [2](https://meps.ahrq.gov/data_stats/download_data/pufs/h192/h192doc.shtml#DataA) |
| **ansur2**                       | Data from the [Open Design Lab](https://www.openlab.psu.edu/)                                                                                                                                                                                              | Public Domain                                                                                                                                                                                                   |
| **births1, births2**             | Data from the [Centers for Disease Control and Prevention](https://data.cdc.gov/) </br> Code for data processing: [lorismichel/drf](https://github.com/lorismichel/drf)                                                                                    |  Data: Public Domain </br> Code: GPL 3.0                                                                                                                                                                                                   |
| **wage**                         | Data from [American Community Survey Data](https://www.census.gov/programs-surveys/acs/data.html) </br> Code for data processing: [lorismichel/drf](https://github.com/lorismichel/drf)                                                                    | Data: Public Domain </br> Code: GPL 3.0                                                                                                                                                                                                   |
| **air**                          | Data from [U.S. Environmental Protection Agency](https://www.epa.gov/)  </br> Code for data processing: [lorismichel/drf](https://github.com/lorismichel/drf)                                                                                              |  Data: Public Domain </br> Code: GPL 3.0                                                                                                                                                                                                   |
| **taxi**                         | Data: [NYC Open Data](https://opendata.cityofnewyork.us/data/)<br/> Code for data processing: [Zhendong-Wang/Probabilistic-Conformal-Prediction](https://github.com/Zhendong-Wang/Probabilistic-Conformal-Prediction/blob/main/pcp/datasets.py)            | Data: Public Domain<br/> Code: MIT                                                                                                                                                                              |
| **calcofi**                      | Data from the [California Cooperative Oceanic Fisheries Investigations](https://calcofi.org/data/data-usage-policy/)                                                                                                                                       | CC BY 4.0                                                                                                                                                                                                       |
| **households**                   | Data: [U.S. Bureau of Labor Statistics (Consumer Expenditure Survey PUMD)](https://www.bls.gov/cex/pumd_data.htm)<br/> Code for data processing: [aschnuecker/Superlevel-sets](https://github.com/aschnuecker/Superlevel-sets)                             | Data: Public Domain<br/> Code: GPL 3.0                                                                                                                                                                          |
| **afhq**                         | Data from [clovaai/stargan-v2](https://github.com/clovaai/stargan-v2) | CC BY-NC 4.0 |
