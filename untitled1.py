import pandas as pd
from constants import constants
import math
import numpy as np
import matplotlib as plt
from scipy.interpolate import interp1d
import os
import pandas as pd
from general import general


df_Vaki_weights_daily_mean = pd.read_csv('data/Preore_Dataset/PREORE_VAKI-Weight_dailymean.csv')
general.interpolate_outliers(df_Vaki_weights_daily_mean['PREORE_VAKI-Weight_dailymean [g]'])
