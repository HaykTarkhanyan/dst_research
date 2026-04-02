import os
import json
import logging
import datetime
import warnings

from pandas.errors import SettingWithCopyWarning

import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from config import *
from DSClassifierMultiQ import DSClassifierMultiQ

import matplotlib.pyplot as plt

from tqdm import tqdm
from time import time

from utils import (calculate_adjusted_density, dbscan_predict,
                   detect_outliers_z_score, evaluate_classifier,
                   evaluate_clustering, filter_by_rule, get_distance,
                   get_kdist_plot, remove_outliers_and_normalize,
                   report_results, run_dbscan)

warnings.simplefilter(action='ignore', category=(SettingWithCopyWarning, FutureWarning, UserWarning))

# ---------------------------- logging ----------------------------

if not os.path.exists(LOG_FOLDER):
    os.mkdir(LOG_FOLDER)

log_file = os.path.join(LOG_FOLDER, "dst.log")

rfh = logging.handlers.RotatingFileHandler(
    filename=log_file,
    mode='a',
    maxBytes=LOGGING_MAX_SIZE_MB*1024*1024,
    backupCount=LOGGING_BACKUP_COUNT,
    encoding=None,
    delay=0
)

logging.getLogger('matplotlib.font_manager').disabled = True

console_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%y-%m-%d %H:%M:%S",
    handlers=[
        rfh,
        console_handler
    ],
)

logger = logging.getLogger(__name__)

class DSEnhanced:
    def __init__(self, method=None, 
                 dataset_folder: str = DATASET_FOLDER, dataset: str = None, datasets_to_exclude: list = None, 
                 nrows: int = None, rows_to_use: int = None,
                 missing_threshold: float = MISSING_THRESHOLD, ratio_deviation: float=RATIO_DEVIATION, 
                 train_set_size: float = TRAIN_SET_SIZE, 
                 eps: float = EPS, step: float = STEP, max_eps: float = MAX_EPS,
                 min_samples: int = None, target_clusters: int = TARGET_CLUSTERS, 
                 print_clustering_eval: bool = True, 
                 print_clustering_as_classification_eval: bool = True, 
                 mult_rules: bool = False, debug_mode: bool = True, print_final_model: bool = True,
                 num_breaks: int = 3, rules_folder: str = RULE_FOLDER, plots_folder: str = PLOTS_FOLDER,
                 run_for_all_mafs: bool = False, methods_lst: list = None, 
                 clustering_alg_list: list = ["kmeans", "dbscan", "means_no_clustering", "density_no_clustering"], 
                 add_in_between_rules: bool = False,
                 res_json_folder: str = None):
        self.clustering_alg_list = clustering_alg_list
        
        self.method = method
        self.methods_lst = methods_lst
        
        if self.methods_lst is None:    
            if run_for_all_mafs:
                self.methods_lst = ["clustering", "random", "uniform"]  
            else:
                if self.method is None:
                    raise Exception("You must specify a method")
                self.methods_lst = [self.method]
        
        # assert self.method in ["clustering", "random", "uniform"], f"Method {self.method} not found in the list of methods"
        
        logging.info(f"Methods list: {self.methods_lst}")                
        
        self.clustering_alg = None
        self.db_eps = None # for density based opacity calculation

        self.eps = eps
        self.step = step
        self.max_eps = max_eps
        if min_samples:
            self.min_samples = min_samples # if not specified, will be assigned later 
        self.target_clusters = target_clusters
        
        self.print_clustering_eval = print_clustering_eval
        self.print_clustering_as_classification_eval = print_clustering_as_classification_eval

            
        self.run_for_all_mafs = run_for_all_mafs
        
        self.dataset_folder = dataset_folder
        assert os.path.exists(dataset_folder), f"Dataset folder `{dataset_folder}` does not exist"

        self.datasets_to_exclude = datasets_to_exclude
        self.datasets = os.listdir(dataset_folder)
        
        logging.info(f"Found {len(self.datasets)} datasets")
        if dataset:
            assert dataset in self.datasets, f"Dataset {dataset} not found in the folder"
            self.dataset = dataset
        else:
            logging.info(f"Using the first dataset: {self.datasets[0]}")
            self.dataset = self.datasets[0]

        if not os.path.exists(rules_folder):
            os.mkdir(rules_folder)
            
        self.rules_folder = rules_folder
        self.plots_folder = plots_folder
        self.dataset_name = self.dataset.split(".")[0]
        self.nrows = nrows
        self.rows_to_use = rows_to_use
        self.ratio_deviation = ratio_deviation
        self.missing_threshold = missing_threshold
        self.train_set_size = train_set_size
                
        # dst
        self.mult_rules = mult_rules
        self.debug_mode = debug_mode
        self.print_final_model = print_final_model
        self.num_breaks = num_breaks
        
        self.clustering_model = None # will get assigned if cl_alg is kmeans or dbscan
        
        self.add_in_between_rules = add_in_between_rules
        
        self.res_json_folder = res_json_folder
        
        self.results = {"durations": {}, "dataset": {}}
        self.results["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def read_data(self):
        st = time()
        if self.nrows:
            self.data_initial = pd.read_csv(os.path.join(self.dataset_folder, self.dataset), 
                                    nrows=self.nrows)
        else:
            self.data_initial = pd.read_csv(os.path.join(self.dataset_folder, self.dataset))
        
        logger.debug(f"Dataset: {self.dataset_name} | Shape: {self.data_initial.shape}")
        
        self.results["durations"]["read_data"] = time() - st
        self.results["dataset"]["name"] = self.dataset_name
        self.results["dataset"]["initial_row_count"] = len(self.data_initial)
        self.results["dataset"]["initial_col_count"] = len(self.data_initial.columns)

    def preprocess_data(self):
        st = time()
        self.data = self.data_initial.dropna(thresh=len(self.data_initial) * (1 - self.missing_threshold), axis=1)
        logger.debug(f"{self.data.shape} - dropped columns with more than {self.missing_threshold*100:.0f}% missing values")
        self.data = self.data.dropna()
        logger.debug(f"{self.data.shape} - drop rows with missing values")
        
        assert self.data.isna().sum().sum() == 0, "Dataset contains missing values"
        assert "labels" in self.data.columns, "Dataset does not contain `labels` column"
        assert self.data.labels.nunique() == 2, f"Dataset labels are not binary ({self.data.labels.unique()})"

        label_ratio = self.data.labels.value_counts(normalize=True).iloc[0]
        assert abs(label_ratio -0.5) < self.ratio_deviation, f"Label ratio is not balanced ({label_ratio})"

        # leave only numeric columns
        self.data = self.data.select_dtypes(include=[np.number])
        logger.debug(f"{self.data.shape} drop non-numeric columns")

        # select only rows_to_use rows but keep the same ratio of labels
        if self.rows_to_use:
            self.data, _ = train_test_split(self.data, train_size=self.rows_to_use / len(self.data),
                                            stratify=self.data['labels'], random_state=42)
            logger.debug(f"{self.data.shape} - select {self.rows_to_use} rows (stratified)")        
        # move labels column to the end 
        self.data = self.data[[col for col in self.data.columns if col != "labels"] + ["labels"]]

        logging.info(f"------ Dataset: {self.dataset_name} | Shape: {self.data.shape} | Label ratio: {label_ratio:.2f} -------")
        
        self.results["durations"]["preprocess_data"] = time() - st
        self.results["dataset"]["label_ratio"] = label_ratio
        self.results["dataset"]["final_row_count"] = len(self.data)
        self.results["dataset"]["final_col_count"] = len(self.data.columns)

    def train_test_split(self):
        st = time()
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
        self.data = self.data.apply(pd.to_numeric)
        cut = int(train_set_size*len(self.data))

        self.train_data_df = self.data.iloc[:cut]
        self.test_data_df = self.data.iloc[cut:]

        self.X_train = self.data.iloc[:cut, :-1].values
        self.y_train = self.data.iloc[:cut, -1].values
        self.X_test = self.data.iloc[cut:, :-1].values
        self.y_test = self.data.iloc[cut:, -1].values

        logging.info(f"Step 0: Data split done | {len(self.X_train)} - {len(self.X_test)}")
        
        self.results["durations"]["train_test_split"] = time() - st
        self.results["dataset"]["train_row_count"] = len(self.X_train)
        self.results["dataset"]["test_row_count"] = len(self.X_test)

    def standard_scaling(self):
        st = time()
        st_scaler = StandardScaler().fit(self.train_data_df)

        # TODO maybe delete this
        # scale = st_scaler.scale_
        # mean = st_scaler.mean_
        # # var = st_scaler.var_ 

        self.X_train_scaled = st_scaler.transform(self.train_data_df)
        self.X_test_scaled = st_scaler.transform(self.test_data_df)
        
        # last column is a binary one, and has gotten scaled, please fix
        self.X_train_scaled[:, -1] = self.train_data_df["labels"].values
        self.X_test_scaled[:, -1] = self.test_data_df["labels"].values
        logging.debug("Step 1: Standard scaling complete")
        
        self.results["durations"]["standard_scaling"] = time() - st

    def clustering_and_inference(self):
        assert self.clustering_alg in ["kmeans", "dbscan"], f"You must specify a clustering algorithm, got {self.clustering_alg}"   
        logging.info(f"Step 2.1: Performing {self.clustering_alg} clustering")

        if self.clustering_alg == "kmeans":
            st_clust = time()
            self.clustering_model = KMeans(n_clusters=2, random_state=42, n_init="auto")      
            self.clustering_model.fit(self.X_train_scaled)  
            
            self.results["durations"]["clustering_fit"] = time() - st_clust
            st_pred = time()

            self.clustering_labels_train = self.clustering_model.predict(self.X_train_scaled)
            self.clustering_labels_test = self.clustering_model.predict(self.X_test_scaled)
            self.results["durations"]["clustering_predict"] = time() - st_pred
            
        elif self.clustering_alg == "dbscan":
            st_clust = time()

            self.min_samples = 2 * self.X_train_scaled.shape[1] - 1
            self.clustering_model = run_dbscan(self.X_train_scaled, eps=self.eps, 
                                               max_eps=self.max_eps, min_samples=self.min_samples, 
                                               step=self.step) 
            if self.clustering_model is None:
                logging.warning(f"Could not find the desired number of clusters for {self.dataset_name}")
                raise Exception("Clustering failed")
            
            self.results["durations"]["clustering_fit"] = time() - st_clust
            
            st_pred = time()
            
            self.clustering_labels_train = dbscan_predict(self.clustering_model, self.X_train_scaled)
            self.clustering_labels_test = dbscan_predict(self.clustering_model, self.X_test_scaled)
            self.results["durations"]["clustering_predict"] = time() - st_pred
            
            self.db_eps = self.clustering_model.eps


        self.train_data_df["labels_clustering"] = self.clustering_labels_train
        self.test_data_df["labels_clustering"] = self.clustering_labels_test

        logger.info(f"Step 2.1: Clustering and inference done")
        
        self.results["clustering_alg"] = self.clustering_alg
        
        
    def run_eval_clustering(self):
        st = time()
        logger.info("Step 2.2: Evaluate clustering")
        self.eval_clustering_train = evaluate_clustering(
            self.X_train_scaled, self.clustering_labels_train, self.clustering_model, 
            self.clustering_alg, print_results=self.print_clustering_eval, dataset="train")
        self.eval_clustering_test = evaluate_clustering(
            self.X_test_scaled, self.clustering_labels_test, self.clustering_model, 
            self.clustering_alg, print_results=self.print_clustering_eval, dataset="test")
        
        self.results["durations"]["clustering_eval"] = time() - st
        self.results["clustering_eval_train"] = self.eval_clustering_train
        self.results["clustering_eval_test"] = self.eval_clustering_test
        logger.info("Step 2.2: Clustering evaluation done")
        
    def run_eval_clustering_as_classifier(self):
        st = time()
        self.eval_clustering_as_classifier_train = evaluate_classifier(
            y_actual=self.y_train, y_clust=self.clustering_labels_train, 
            dataset="train", print_results=self.print_clustering_as_classification_eval)
        self.eval_clustering_as_classifier_test = evaluate_classifier(
            y_actual=self.y_test, y_clust=self.clustering_labels_test, 
            dataset="test", print_results=self.print_clustering_as_classification_eval)

        self.results["durations"]["clustering_as_classifier_eval"] = time() - st
        self.results["clustering_as_classifier_eval_train"] = self.eval_clustering_as_classifier_train
        self.results["clustering_as_classifier_eval_test"] = self.eval_clustering_as_classifier_test
        logger.info("Step 3: Clustering as a classifier, evaluation done")

    def get_opacity(self):
        st = time()
        self.train_data_df["distance"] = get_distance(
            self.X_train_scaled, self.clustering_model, 
            self.clustering_alg, density_radius=self.db_eps)
        self.test_data_df["distance"] = get_distance(
            self.X_test_scaled, self.clustering_model, 
            self.clustering_alg, density_radius=self.db_eps)

        self.train_data_df["distance_norm"] = remove_outliers_and_normalize(
            self.train_data_df) 
        self.test_data_df["distance_norm"] = remove_outliers_and_normalize(
            self.test_data_df)

        assert self.train_data_df.isna().sum().sum() == 0, "Train data contains NaNs"
        assert self.test_data_df.isna().sum().sum() == 0, "Train data contains NaNs"

        self.results["durations"]["opacity_calculation"] = time() - st
        logger.info(f"Step 4: Opacity calculation done")
        
    def train_DST(self):
        ignore_for_training = ["labels_clustering", "distance_norm"]
        df_cols = [i for i in list(self.data.columns) if i not in ignore_for_training]

        logger.debug(f"Train: {len(self.X_train)}")

        name = f"dataset={self.dataset_name}, label_for_dist={LABEL_COL_FOR_DIST}, clust={self.clustering_alg}, breaks={self.num_breaks}, add_mult_rules={self.mult_rules}, maf_method={self.method}, in_between_rules={self.add_in_between_rules}"
        logger.info(f"Step 5: Run DST ({name})")
        st = time()
        DSC = DSClassifierMultiQ(2, debug_mode=self.debug_mode, num_workers=0, maf_method=self.method,
                                data=self.train_data_df, precompute_rules=True, 
                                add_in_between_rules=self.add_in_between_rules)#.head(rows_use))
        logger.debug(f"\tModel init done")    
        res = DSC.fit(self.X_train, self.y_train, 
                add_single_rules=True, single_rules_breaks=self.num_breaks, add_mult_rules=self.mult_rules,
                column_names=df_cols, print_every_epochs=1, print_final_model=self.print_final_model)
        losses, epoch, dt = res
        logger.debug(f"\tModel fit done")
        
        self.results["durations"]["dst_fit"] = time() - st

        if not os.path.exists(os.path.join(self.rules_folder, self.dataset_name)):
            os.mkdir(os.path.join(self.rules_folder, self.dataset_name))
        rules = DSC.model.save_rules_bin(os.path.join(self.rules_folder, self.dataset_name, f"{name}.dsb"))

        self.rules = DSC.model.find_most_important_rules()
        st = time()
        y_train_pred = DSC.predict(self.X_train)
        y_test_pred = DSC.predict(self.X_test)
        
        
        self.results["durations"]["dst_predict"] = time() - st

        logger.info(f"Step 6: Inference done")

        self.dst_res = report_results(self.y_train, y_train_pred, self.y_test, y_test_pred, dataset=self.dataset_name, method=self.method,
                    epoch=epoch, dt=dt, losses=losses, 
                    save_results=True, name=name, print_results=True,
                    breaks=self.num_breaks, mult_rules=self.mult_rules, 
                    clustering_alg=self.clustering_alg, label_for_dist=LABEL_COL_FOR_DIST, plot_folder=self.plots_folder)
        logging.info("-"*30)
        
        self.results["dst_results"] = self.dst_res
        self.results["durations"]["dst"] = time() - st
        self.results["dst_results"]["rules"] = DSC.model.get_rules_size()
        self.results["added_in_between_rules"] = self.add_in_between_rules
        

    # def save_all_important_res_to_json(self):
    #     logging.info("Step 7: Save all important results to json")
    #     for attr in ["eval_clustering_train", "eval_clustering_test", "eval_clustering_as_classifier_train", 
    #                  "eval_clustering_as_classifier_test", "opacity_train", "opacity_test", "dataset", 
    #                  "clustering_alg", "num_breaks", "mult_rules", "maf_methods", "train_data", "test_data", 
    #                  "rules", "X_train", "y_train", "X_test", "y_test", "X_train_scaled", "X_test_scaled", 
    #                  "clustering_labels_train", "clustering_labels_test", "db_eps", "train_data_df_use", 
    #                  "test_data_df_use", "X_train_use", "y_train_use", "X_test_use", "y_test_use", 
    #                  "train_data_df", "test_data_df", "data", "data_initial", "train_data_df_use", 
    #                  "test_data_df_use", "X_train_use", "y_train_use", "X_test_use", "y_test_use", 
    #                  "train_data_df", "test_data_df", "X_train_scaled", "X_test_scaled", 
    #                  "clustering_labels_train", "clustering_labels_test", "db_eps", "train_data_df_use", 
    #                  "test_data_df_use", "X_train_use", "y_train_use"]:
    #         self.dst_res[attr] = getattr(self, attr)
    
    def run(self):
        self.read_data()
        self.preprocess_data()
        self.train_test_split()
        for method in tqdm(self.methods_lst):   
            self.clustering_alg = None
 
            self.method = method
            logging.info(f"----------- Running {self.method} method -----------")

            if self.method == "clustering":
                self.standard_scaling()
                for cl_alg in self.clustering_alg_list:
                    logging.info(f"----------- Running {cl_alg} clustering -----------")
                    self.clustering_alg = cl_alg

                    if cl_alg in ["kmeans", "dbscan"]:
                        self.clustering_and_inference()
                        self.run_eval_clustering()
                        self.run_eval_clustering_as_classifier()
                        self.get_opacity()  
                        self.train_DST()
                    elif cl_alg in ["means_no_clustering", "density_no_clustering"]:
                        self.get_opacity()  
                        self.train_DST()
                    else: 
                        raise Exception(f"Clustering algorithm {cl_alg} not found")
                    self.save_results()
            elif self.method in ["random", "uniform"]:
                self.train_DST()
                self.save_results()
            else:
                raise Exception(f"Method {self.method} not found")  
                

            logging.info(f"Finished {self.dataset_name}")
        logging.info("Finished all MAF methods")
        

    def run_all_datasets(self):
        if self.datasets_to_exclude:
            self.datasets = [dataset for dataset in self.datasets 
                             if dataset not in self.datasets_to_exclude]
        
        for dataset in tqdm(self.datasets):
            self.dataset = dataset
            self.dataset_name = self.dataset.split(".")[0]
            self.run()

        logging.info("Finished all datasets")
        
    def save_results(self):
        # there may be multiple nestings
        for key, value in self.results.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        self.results[key][k] = v.tolist()
                    elif isinstance(v, plt.Figure):
                        self.results[key][k] = "todo"#v.to_json()                                
                                    
                    elif isinstance(v, dict):
                        for kk, vv in v.items():
                            if isinstance(vv, np.ndarray):
                                self.results[key][k][kk] = vv.tolist()
                            elif isinstance(vv, plt.Figure):
                                self.results[key][k][kk] = "todo"#vv.to_json()
                            elif isinstance(vv, dict):
                                for kkk, vvv in vv.items():
                                    if isinstance(vvv, np.ndarray):
                                        self.results[key][k][kk][kkk] = vvv.tolist()
                                    elif isinstance(vvv, plt.Figure):
                                        self.results[key][k][kk][kkk] = "todo"#vvv.to_json()
        
        if not os.path.exists(self.res_json_folder):
            os.mkdir(self.res_json_folder)
            
        
        folder_name = os.path.join(self.res_json_folder, self.dataset_name)                 
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        
        with open(os.path.join(folder_name, f"{self.dataset_name}_{self.method}_{self.clustering_alg}_extra_rules_{self.add_in_between_rules}.json"), "w") as f:
            json.dump(self.results, f, indent=4)
    
    