import pandas as pd
import numpy as np
import glob
import re
from subprocess import PIPE, Popen

SAR_NODE_NAMES = ['node1', 'node2', 'node4', 'node5', 'node6', 'node7', 'node8', 'node9']
SAR_LOG_DIR = '/home/ceph-user/sar-logs'
list_of_files = glob.glob(SAR_LOG_DIR + '/*')

print(len(list_of_files))

collected_rtps = []
collected_wtps = []

rtps_dict = dict()
wtps_dict = dict()

total = 1500

for k in SAR_NODE_NAMES:
    rtps_dict[k] = np.zeros(total)
    wtps_dict[k] = np.zeros(total)

counts = {k: 0 for k in SAR_NODE_NAMES}

for i, file in enumerate(list_of_files):
    print(i)
    # get the all lines
    raw_data = Popen(['cat', file], shell=False, stdout=PIPE)  # update 11/30
    data = raw_data.stdout.read().strip()
    data = str(data)

    exp4 = r'(node[1-9]):( )+[0-9][0-9]:[0-9][0-9]:[0-9][0-9]( )+[AP][M]( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)'
    finds4 = re.findall(exp4, data)

    new_data4 = []
    for f in finds4:
        dat4 = []
        number4 = 0
        for d in f:
            if d.startswith('node'):
                number4 = d[-1]
            elif d != ' ':
                dat4.append(d)

        new_data4.append((number4, dat4))

    new_data4 = sorted(new_data4, key=lambda t: t[0])
    # print(new_data4)
    #
    # rtps_list = []
    # wtps_list = []

    for k, v in new_data4:
        rtps = float(v[1])
        wtps = float(v[2])
        name = 'node{}'.format(k)
        rtps_dict[name][counts[name]] = rtps
        wtps_dict[name][counts[name]] = wtps
        counts[name] += 1


        #
        # # TODO
        # rtps_list.append(rtps)  # normalization factor divide by max, write a script
        # wtps_list.append(wtps)

    # rtps_array = np.array(rtps_list)
    # wtps_array = np.array(wtps_list)
    # # print(rtps_array.shape)
    # collected_rtps.append(rtps_array)
    # collected_wtps.append(wtps_array)
# print(rtps_dict.keys())
pd.DataFrame.from_dict(rtps_dict).to_csv("rtps_data.csv", index=False)
pd.DataFrame.from_dict(wtps_dict).to_csv("wtps_data.csv", index=False)

