import logging

cvss_task_type = {
    "cvss2_C": {"P": 0, "N": 1, "C": 2},
    "cvss2_I": {"P": 0, "N": 1, "C": 2},
    "cvss2_A": {"P": 0, "N": 1, "C": 2},
    "cvss2_AV": {"N": 0, "L": 1, "A": 2},
    "cvss2_AC": {"L":0,"M":1,"H":2},
    "cvss2_AU": {"S":0,"N":1} ,
    "cvss2_severity" : {"LOW":0,"MEDIUM":1,"HIGH":2},
}

n_gram_vocab_size = 250499


def init_logging(logFilename):
    ''' Output log to file and console '''
    # Define a Handler and set a format which output to file
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s: %(message)s',
        datefmt='%m-%d %H:%M:%S',
        filename=logFilename,  # log file name
        filemode='w')
    console = logging.StreamHandler()  # define console handler
    console.setLevel(logging.INFO)  # handler level
    formatter = logging.Formatter('%(asctime)s: %(message)s',datefmt='%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

def log_info(msg:str):
    logging.info(msg)