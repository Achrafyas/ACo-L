# Config XMPP Server
# xmpp_server = "gtirouter.dsic.upv.es"
xmpp_server = "localhost"

# Web
web_port = 5222
url = "localhost"

# Load model path = False -> overwrites the model from scratch
load_model_path = False

# Export the logs and clean log directories
export_logs_root_path = "../experiment_logs/"
export_logs_folder_prefix = "exp"
export_logs_at_end_of_execution = False
export_logs_append_to_last_log_folder_instead_of_create = True  # True: Requires manually creating the folder after an experiment, to avoid the mixing
logs_root_folder = "Logs"
logs_folders = [
    "Epsilon Logs",
    "Message Logs",
    "Training Logs",
    "Training Time Logs",
    "Weight Logs",
]

# Downloaded dataset folder
data_set_path = "data/datasets"

# Max SPADE message body length (aioxmpp limit is 256 * 1024)
max_message_body_length = 150_000

# Max seconds to accept a response and a pre-consensus message
max_seconds_timeout_response = 20  # FSM
max_seconds_pre_consensus_message = 30  # Cyclic

# Coalition properties
coalition_probability = -1  # -1: ACoL; >0: ACoaL
coalitions = [["0", "1", "2", "3"], ["4", "5", "6", "7"]]
max_training_iterations = 120  # -1 : infinite

iid = False  # IDEAL2024

acol_mimics_acoal_distribution = False

cifar8_use_full_test_data_when_coalitions = False
cifar8_coalitions_iid_all_the_data = False  # deprecated

# CIFAR4 distribution of samples (keys -1, 0, 1 are (-1 = acol) acol and (0 = coalition0, 1 = coalition1) coalition indexes)
cifar4_dataset_distribution = {
    "train": {
        -1: {"cats": 2500, "dogs": 2500, "deers": 2500, "horses": 2500},
        0: {"cats": 5000, "dogs": 5000, "deers": 0, "horses": 0},
        1: {"cats": 0, "dogs": 0, "deers": 5000, "horses": 5000},
    },
    "test": {
        -1: {"tst_cats": 500, "tst_dogs": 500, "tst_deers": 500, "tst_horses": 500},
        0: {"tst_cats": 1000, "tst_dogs": 1000, "tst_deers": 0, "tst_horses": 0},
        1: {"tst_cats": 0, "tst_dogs": 0, "tst_deers": 1000, "tst_horses": 1000},
    },
}

fruit4_dataset_distribution = {
    "train": {
        -1: {
            "Apple Braeburn": 0.8,
            "Apple Granny Smith": 0.8,
            "Blueberry": 0.8,
            "Grape Blue": 0.8,
        },
        0: {
            "Apple Braeburn": 1,
            "Apple Granny Smith": 1,
            "Blueberry": 0,
            "Grape Blue": 0,
        },
        1: {
            "Apple Braeburn": 0,
            "Apple Granny Smith": 0,
            "Blueberry": 1,
            "Grape Blue": 1,
        },
    }
}


# FSM
SETUP_STATE_AG = "SETUP_STATE"
RECEIVE_STATE_AG = "RECEIVE_STATE"
TRAIN_STATE_AG = "TRAIN_STATE"
SEND_STATE_AG = "SEND_STATE"
TEST_STATE_AG = "TEST_STATE"
RESPOND_STATE_AG = "RESPOND_STATE_AG"
IDLE_STATE_AG = "IDLE_STATE_AG"

# LOGGERS
CONSENSUS_LOGGER = "CONSENSUS_LOGGER"
MESSAGE_LOGGER = "MESSAGE_LOGGER"
WEIGHT_LOGGER = "WEIGHT_LOGGER"
TRAINING_LOGGER = "TRAINING_LOGGER"
EPSILON_LOGGER = "EPSILON_LOGGER"
TRAINING_TIME_LOGGER = "TRAINING_TIME_LOGGER"