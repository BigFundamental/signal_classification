#! -*- encoding: utf8 -*-

import pandas as pd
import numpy as np
from filter import Filter
from feature_extractor import FeatureExtractor
import logging

logger = logging.getLogger('server')

ModelVersionFeatureConfig = {
        "ada_0830_B" : {"features": ['peaks_num', 'up_edges_num', 'down_edges_num', 'down_peaks_num', 'peak_edge_ratio', 'down_peak_edge_ratio', 'edge_diff_10', 'edge_diff_20', 'edge_diff_50', 'width_diff_10'], "path": "model.ada.2018.06.04"},
        "gbdt_0830_B" : {"features": ['peaks_num', 'up_edges_num', 'down_edges_num', 'down_peaks_num', 'peak_edge_ratio', 'down_peak_edge_ratio', 'edge_diff_10', 'edge_diff_20', 'edge_diff_50', 'width_diff_10', 'negative_peak_num'], "path": "model.gbdt.2018.06.08"},
        "gbdt_0830_BA" : {"features": ['peaks_num', 'up_edges_num', 'down_edges_num', 'down_peaks_num', 'peak_edge_ratio', 'down_peak_edge_ratio', 'edge_diff_10', 'edge_diff_20', 'width_diff_10', 'negative_peak_num', 'max_down_peak_point'], "path": "model.gbdt.2018.06.10"},
        "gbdt_0830BA_SKEW_LS": {"features": ['peaks_num', 'down_peaks_num', 'up_edges_num', 'down_edges_num','peak_edge_ratio', 'down_peak_edge_ratio', 'edge_diff_10', 'edge_diff_20', 'width_diff_10', 'negative_peak_num', 'max_down_peak_point'], "path": "model.gbdt.2019.06.18"},
        "test": {"features": ['peaks_num', 'up_edges_num', 'down_edges_num', 'down_peaks_num', 'peak_edge_ratio', 'down_peak_edge_ratio', 'edge_diff_10', 'edge_diff_20', 'width_diff_10', 'negative_peak_num', 'max_down_peak_point'], "path": "model.test"}
}
