import numpy as np
import random
import tensorflow as tf
import pandas as pd
from disusedModel import ONEDCNN_NB15
import model
from scipy.stats import chi2_contingency
import os


def readData(file):
    raw_data = pd.read_csv(file, header=None, low_memory=False)
    return raw_data.drop([0])


def mergeData():
    monday = readData("D:/CIC-IDS2017Dataset/csv/Monday-WorkingHours.pcap_ISCX.csv")
    friday1 = readData("D:/CIC-IDS2017Dataset/csv/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    friday2 = readData("D:/CIC-IDS2017Dataset/csv/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
    friday3 = readData("D:/CIC-IDS2017Dataset/csv/Friday-WorkingHours-Morning.pcap_ISCX.csv")
    thursday1 = readData("D:/CIC-IDS2017Dataset/csv/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
    thursday2 = readData("D:/CIC-IDS2017Dataset/csv/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
    tuesday = readData("D:/CIC-IDS2017Dataset/csv/Tuesday-WorkingHours.pcap_ISCX.csv")
    wednesday = readData("D:/CIC-IDS2017Dataset/csv/Wednesday-workingHours.pcap_ISCX.csv")
    frame = [monday, friday1, friday2, friday3, thursday1, thursday2, tuesday, wednesday]
    result = pd.concat(frame)
    drop_list = clearDirtyData(result)
    result = result.drop(drop_list)
    return result


def clearDirtyData(df):
    dropList = df[(df[14] == "Nan") | (df[15] == "Infinity")].index.tolist()
    return dropList


if __name__ == '__main__':
    raw = mergeData()
    # last_column_index = raw.shape[1] - 1
    # print(raw[last_column_index].value_counts())
    raw.to_csv("D:/CIC-IDS2017Dataset/concat.csv", index=False, columns=None, header=False)
