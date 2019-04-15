VMSTAT_NODE_NAMES = ['node1', 'node2', 'node4',
                     'node5', 'node6', 'node7', 'node8', 'node9']
SYSBENCH_NODE_NAMES = ['node1', 'node2', 'node4',
                       'node5']
ALL_NODE_NAMES = ['cosbench', 'admin'] + VMSTAT_NODE_NAMES
NUM_NODES = len(VMSTAT_NODE_NAMES)  # cluster size excluding master node
VMSTAT_LOGS_DIR = '/home/ceph-user/vmstat_logs'
COSBENCH_LOGS_DIR = '/home/ceph-user/cos/archive'
PHANTOMJS_PATH = '/usr/local/share/phantomjs-2.1.1-linux-x86_64/bin/phantomjs'
COSBENCH_WEBGUI_URL_BASE = 'http://129.114.33.85:19088/controller/workload.html?id='
TOTAL_MEM = 4048288
NUM_STATS = 3  # number of system stats to consider
WAIT = 20  # in seconds
