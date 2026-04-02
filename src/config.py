LOWER_CONFIDENCE_BY_PROPORTION = True
OUTLIER_THRESHOLD_NUM_STD = 2

# DSClassifierMultiQ
max_iter = 500

# DBSCAN
# Initialization
EPS = 0.01  # Initial epsilon value
STEP = 0.05  # Step size for epsilon increment
MAX_EPS = 20  # Maximum epsilon boundary
MIN_SAMPLES = 2  # You can adjust this based on data density
TARGET_CLUSTERS = 2  # Desired number of clusters excluding noise

train_set_size = 0.7
# https://www.kaggle.com/datasets/abhinav89/telecom-customer

LOGGING_MAX_SIZE_MB = 5
LOGGING_BACKUP_COUNT = 2

# confidence calculation dbscan
penalty_rate = 0.5

print_results_MAF_kmeans = False

LABEL_COL_FOR_DIST = "labels_clustering"
LABEL_COL_FOR_DIST = "labels"

LOG_FOLDER = "logs"
DATASET_FOLDER = "datasets"
RULE_FOLDER = "rules"
PLOTS_FOLDER = "plots"

MISSING_THRESHOLD = 0.2 # will remove columns with more than 20% missing values
RATIO_DEVIATION = 0.4 # acceptable deviation of class balance (0.4 means 0.1-0.9)
TRAIN_SET_SIZE = 0.7 
