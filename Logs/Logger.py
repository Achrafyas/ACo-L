import datetime
import Config

class Logger:
    """
    Simplest CSV logger: writes a header based on logger_type,
    then appends timestamped lines via write_to_file().
    """

    def __init__(self, file_name, logger_type):
        """
        Create (or overwrite) the log file and write its CSV header.
        :param file_name: Path to the CSV log file.
        :param logger_type: One of the Config.*_LOGGER constants.
        """
        self.file_name = file_name
        # Open in write mode to reset/initialize the file
        with open(file_name, "w") as f:
            # Choose header based on the type of data to log
            if logger_type == Config.CONSENSUS_LOGGER:
                # Columns: time, which weight was received, who sent it, resulting weight
                f.write("time,received_weight,sending_agent,weight\n")
            elif logger_type == Config.MESSAGE_LOGGER:
                # Columns: time, send/receive, message ID, peer agent
                f.write("time,send_or_recv,id,communicating_agent\n")
            elif logger_type == Config.TRAINING_LOGGER:
                # Columns: time, train_acc, train_loss, test_acc, test_loss
                f.write("time,training_accuracy,training_loss,test_accuracy,test_loss\n")
            elif logger_type == Config.WEIGHT_LOGGER:
                # Columns: time, phase, first-layer weights & biases, second-layer weights & biases
                f.write(
                    "time,train_or_consensus,"
                    "first_layer_weight,first_layer_bias,"
                    "second_layer_weight,second_layer_bias\n"
                )
            elif logger_type == Config.EPSILON_LOGGER:
                # Columns: time, epsilon value
                f.write("time,value\n")
            elif logger_type == Config.TRAINING_TIME_LOGGER:
                # Columns: time, marker ("START" or "STOP")
                f.write("time,start_or_stop\n")

    def write_to_file(self, content):
        """
        Append a new line to the CSV, prefixed with the current timestamp.
        :param content: Comma-separated values matching the header (excluding time).
        """
        timestamp = datetime.datetime.now()
        with open(self.file_name, "a") as f:
            f.write(f"{timestamp},{content}\n")