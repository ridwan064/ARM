"""An abstract class for Interaction such as parsing as well as changing
status/properties of the cluster"""

# env req
import numpy as np
import os
from datetime import datetime
from subprocess import Popen, PIPE
from selenium import webdriver
from bs4 import BeautifulSoup
import glob
import time
import re

"""-------1. HELPER PARAMS-------"""

SAR_NODE_NAMES = ['node1', 'node2', 'node4', 'node5', 'node6', 'node7', 'node8', 'node9']
MAXES = [2200, 1800, 320, 210]  # rhs, whs, rhl, whl
ALL_NODE_NAMES = ['cosbench', 'admin'] + SAR_NODE_NAMES
NUM_NODES = len(SAR_NODE_NAMES)  # cluster size excluding master node
SAR_LOG_DIR = '/home/ceph-user/sar-logs'
COSBENCH_LOGS_DIR = '/home/ceph-user/cos/archive'
PHANTOMJS_PATH = '/usr/local/share/phantomjs-2.1.1-linux-x86_64/bin/phantomjs'
COSBENCH_WEBGUI_URL_BASE = 'http://129.114.111.174:19088/controller/workload.html?id='
TOTAL_NET = 458752  # KB, assuming maximum network bandwidth = 3.5Gbps = 458752KBps or 19200 KB if we assume max network bandwidth = 150Mbps (50 threads in iperf)

TOTAL_STATS = 7  # [cpu_usage_array, net_usage_array, io_wait_array, rtps_array, wtps_array, affinity_array, reweight_array]
USEFUL_STATS = 5  # [cpu_usage_array, net_usage_array, io_wait_array, rtps_array, wtps_array]

WAIT = 20  # in seconds
N_SLOTS = 10
"""-------2. INTERFERENCE FUNCTIONS-------"""


# TODO: randomize with 2(one of the intereferece) and 2(another one) sysbench nodes

def execute_cpu(node_names_list):

    node_names = ",".join(node_names_list)
    cpu_interference = "pdsh -w {} 'sysbench --num-threads=256 --test=cpu --cpu-max-prime=500000 run --max-time=400s'".format(
        node_names)
    cpu_interference = "({};{})&".format(cpu_interference, cpu_interference)

    print('Running sysbench with cmd: {}'.format(cpu_interference))
    p3 = Popen(cpu_interference, shell=True)
    p3.communicate()
    p3.wait()


def execute_io(node_names_list):

    node_names = ",".join(node_names_list)
    io_interference = "pdsh -w {} 'cd /home/ceph-user/fio-data;fio --name=randread --ioengine=libaio --iodepth=16 --rw=randread --bs=4k --direct=0 --size=2G --numjobs=4 --runtime=400'".format(
        node_names)
    io_interference = "({};{})&".format(io_interference, io_interference)
    print('Running fio with cmd: {}'.format(io_interference))
    p3 = Popen(io_interference, shell=True)
    p3.communicate()
    p3.wait()


def execute_net(node_names_list):

    net1 = "pdsh -w {} 'iperf -s -w 4M' &".format(node_names_list[0])
    net2 = "pdsh -w {} 'iperf -c {} -w 8M -t 400 -P 50' &".format(node_names_list[1], node_names_list[0])
    print('Running iperf with cmd: {}'.format(net1))
    p3 = Popen(net1, shell=True)
    p3.communicate()
    p3.wait()
    print("Net iperf sleep 5s...")
    time.sleep(5)
    print('Running iperf with cmd: {}'.format(net2))
    p3 = Popen(net2, shell=True)
    p3.communicate()
    p3.wait()


"""-------3. INTERACT  HELPER CLASS-------"""


class ClusterEnvironment:
    def __init__(self):
        """All the class variables that might be required in the cluster
        declared here."""
        self.USEFUL_STATS = USEFUL_STATS + 1  # one extra for using affinity
        self.total_net = TOTAL_NET
        self.sar_node_names = SAR_NODE_NAMES
        self.node_num = NUM_NODES
        self.all_nodes = ALL_NODE_NAMES
        self.sar_log_dir = SAR_LOG_DIR
        self.cosbench_log_folder = COSBENCH_LOGS_DIR
        self.phantomjs_path = PHANTOMJS_PATH
        self.cosbench_url = COSBENCH_WEBGUI_URL_BASE
        self.maxes = MAXES # rhs, whs, rhl, whl
        self.n_slots = N_SLOTS
        self.interferences = [execute_cpu, execute_io, execute_net]
        self.train_workloads = ["workload1.xml", "workload2.xml", "workload3.xml", "workload4.xml"]
        self.step_count = 0
        self.start_cluster()

    def start_cluster(self):
        """
        THIS IS THE ENTRY POINT FOR ALL BENCHMARKS RUN: all setup processes
        for cluster done here
        :return: True when ready for agent to start working
        """

        print("Starting cluster...")
        timestamp = '-'.join(datetime.today().__str__().split())[:-7]
        extnsn = 'cos_h'
        new_filename = '{}.{}'.format(timestamp, extnsn)

        # OBSERVATIONS: SAR
        sar_node_names = ",".join(self.sar_node_names)
        sar_cmd = "pdsh -w {} 'sar -b -u -r -n DEV 20 50' > {}/{} &".format(sar_node_names, self.sar_log_dir, new_filename) # update 11/30

        print('Running sar with cmd: {}'.format(sar_cmd))
        p1 = Popen(sar_cmd, shell=True)
        p1.communicate()
        p1.wait()
        print('Sleeping for 40s..Else no obs data..ERROR in agent: SHAPE')
        time.sleep(40)

        # REWARDS: COSBENCH
        random_workload = np.random.choice(self.train_workloads, replace=False)
        cosbench_cmd = "pdsh -w cosbench 'cd /home/ceph-user/cos; sh cli.sh submit conf/{}' &".format(random_workload)
        print('Running cosbench with cmd: {}'.format(cosbench_cmd))
        p2 = Popen(cosbench_cmd, shell=True)
        p2.communicate()
        p2.wait()

        time.sleep(3)

        # RANDOM INTERFERENCES
        random_nodes = list(np.random.choice(self.sar_node_names, size=4, replace=False))
        random_interference = np.random.choice(self.interferences, size=2, replace=False)
        random_interference[0](random_nodes[:2])
        random_interference[1](random_nodes[2:])

        return True

    def stop_cluster(self):
        """stop every task related processes on cluster, bring them to default state
        :return: True when done"""

        # KILL: INTERFERENCES & SAR
        kills = ["sysbench", "fio", "iperf", "sar"]

        for kill in kills:
            kill_command = "pdsh -w {} 'pkill -9 {}'".format(','.join(self.sar_node_names), kill)
            print('Killing {} with cmd: {}'.format(kill, kill_command))
            p = Popen(kill_command, shell=True)
            p.communicate()
            p.wait()

        # KILL: COS BENCH

        # get running job id
        cosbench_info_command = "pdsh -w cosbench 'sh /home/ceph-user/cos/cli.sh info'"
        p3 = Popen(cosbench_info_command, stdout=PIPE, shell=True)
        text = p3.stdout.read().strip()
        text = str(text).strip()
        p3.communicate()
        p3.wait()
        # print (text)
        # job_id = text.splitlines()[5].strip().split(":")[1].strip().split()[0].strip()
        wexp = r': (w[0-9]+)'
        finds = re.findall(wexp, text)
        found_job = len(finds) != 0
        if found_job:
            print("jobs found ", len(finds))
            for job_id in finds:
                cosbench_kill_cmd = "pdsh -w cosbench 'sh /home/ceph-user/cos/cli.sh cancel {}'".format(
                    job_id)
                print('Killing Cosbench Job: {}'.format(job_id))
                print('Killing Cosbench with cmd: {}'.format(cosbench_kill_cmd))
                p4 = Popen(cosbench_kill_cmd, shell=True)
                p4.communicate()
                p4.wait()
                print('Killed Cosbench Job: {}'.format(job_id))
        else:
            print("All set no cosbench Job is running..")

        conts = ["mycontainers1", "mycontainers2"]
        time.sleep(5)
        for c in conts:
            clear_cont = "rados -p {} cleanup --prefix my".format(c)
            print('Clearing {} with cmd: {}'.format(c, clear_cont))
            pc = Popen(clear_cont, shell=True)
            pc.communicate()
            pc.wait()
            time.sleep(5)

        # Sleep ?
        print('Sleep for 15s..')
        time.sleep(15)
        clear_cache_cmd = "pdsh -w {} 'sync && echo 3 | sudo tee /proc/sys/vm/drop_caches'".format(','.join(self.all_nodes))

        print('Clearing cache with cmd: {}'.format(clear_cache_cmd))
        p5 = Popen(clear_cache_cmd, shell=True)
        p5.communicate()
        p5.wait()

        print('Killing Zombie PhantomJS Process...')

        # kill_pjs = "kill -9 `ps -ef | grep phantomjs  | grep -v grep | awk '{print $2}'`"
        kill_pjs = "pkill -9 phantomjs"
        p6 = Popen(kill_pjs, stdout=PIPE, shell=True)
        p6.communicate()
        p6.wait()

        print('Stopped Cluster...')
        return True

    def get_affinities(self):
        """get affinities of the cluster at any given time
        :return: a numpy 1D vector of shape (num_nodes, 1) containing
        affinities of the given cluster"""

        command = ['ceph', 'osd', 'tree']
        p = Popen(command, stdout=PIPE)
        text = p.stdout.read()
        p.communicate()
        p.wait()
        affinity_list = []
        n = 20
        i = 0
        while n < (20 + self.node_num * 10) and i < self.node_num:
            affinity_list.insert(i, text.split()[n])
            n = n + 10
            i = i + 1
        affinity_array = np.array(affinity_list, dtype=np.float32)
        return affinity_array

    def get_weights(self):
        command = ['ceph', 'osd', 'tree']
        p = Popen(command, stdout=PIPE)
        text = p.stdout.read()
        p.communicate()
        p.wait()
        weight_list = []
        n = 19
        i = 0
        while n < (20 + self.node_num * 10) and i < self.node_num:
            weight_list.insert(i, text.split()[n])
            n = n + 10
            i = i + 1
        weight_array = np.array(weight_list, dtype=np.float32)
        return weight_array

    # OBSERVATIONS
    def get_sar_all_nodes(self):
        """to get SAR scores as well as affinity of self.step_count timestep
        :return: a numpy ndarray with shape (num_nodes, USEFUL_STATS) where the
        0th column is affinities"""

        affinity_array = self.get_affinities()
        weight_array = self.get_weights()
        # get latest SAR log file
        list_of_files = glob.glob(SAR_LOG_DIR + '/*')
        latest_file = max(list_of_files, key=os.path.getctime)

        # get the last 104 lines for 8 nodes
        raw_data = Popen(['tail', '-n', '104', latest_file], shell=False, stdout=PIPE) # update 11/30
        data = raw_data.stdout.read().strip()
        data = str(data)

        # cpu + io data parsing
        exp1 = r'(node[1-9]):( )+[0-9][0-9]:[0-9][0-9]:[0-9][0-9]( )+[AP][M]( )+[a][l][l]( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)'
        finds1 = re.findall(exp1, data)

        new_data1 = []
        for f in finds1:
            dat1 = []
            number1 = 0
            for d in f:
                if d.startswith('node'):
                    number1 = d[-1]
                elif d != ' ':
                    dat1.append(d)

            new_data1.append((number1, dat1))
        new_data1 = sorted(new_data1, key=lambda t: t[0])

        cpu_usage_list = []
        io_wait_list = []

        for k, v in new_data1:
            cpu1 = float(v[0])
            cpu2 = float(v[2])
            io = float(v[3])

            cpu_usage_list.append((cpu1 + cpu2) / 100)
            io_wait_list.append(io / 100)

        cpu_usage_array = np.array(cpu_usage_list)
        io_wait_array = np.array(io_wait_list)

        # network data parsing
        exp3 = r'(node[1-9]):( )+[0-9][0-9]:[0-9][0-9]:[0-9][0-9]( )+[AP][M]( )+[e][t][h][0]( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)'
        finds3 = re.findall(exp3, data)
        new_data3 = []
        for f in finds3:
            dat3 = []
            number3 = 0
            for d in f:
                if d.startswith('node'):
                    number3 = d[-1]
                elif d != ' ':
                    dat3.append(d)

            new_data3.append((number3, dat3))
        new_data3 = sorted(new_data3, key=lambda t: t[0])

        net_usage_list = []
        total_net = self.total_net
        for k, v in new_data3:
            rxkb = float(v[2])
            txkb = float(v[3])
            net = rxkb / total_net if rxkb > txkb else txkb / total_net
            net_usage_list.append(net)

        net_usage_array = np.array(net_usage_list)

        # rtps + wtps data parsing - update 11/30

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

        rtps_list = []
        wtps_list = []

        for k, v in new_data4:
            rtps = float(v[1])
            wtps = float(v[2])

            # TODO
            rtps_list.append(rtps / 2000)   # normalization factor divide by experimental max
            wtps_list.append(wtps / 2000)

        rtps_array = np.array(rtps_list)
        wtps_array = np.array(wtps_list)

        obs = np.array([cpu_usage_array, net_usage_array, io_wait_array, rtps_array, wtps_array, affinity_array, weight_array])  # affinity, weight is just there, will not be used
        return obs

    def parse_html_for_cosbench(self, url):
        # get RTs from web UI
        driver = webdriver.PhantomJS(executable_path=self.phantomjs_path)
        driver.get(url)
        html = driver.page_source

        soup = BeautifulSoup(html, 'html.parser')
        try:
            table = soup.find('table', attrs={'class': 'info-table'})

            rows = []
            # read index = 0, write index = 1
            for row in table.find_all("tr")[1:]:
                datapoints = [td.get_text().strip()
                              for td in row.find_all("td")]
                rows.append(datapoints)

        except AttributeError as e:
            print(e)
            print('Retrying Parsing...')
            time.sleep(3)
            rows = self.parse_html_for_cosbench(url)

        print('Number of rows found, 3 total for 2nd stage: {}/3'.format(len(rows)))
        print('Cosbench Data: ', rows)

        return rows

    def check_get_rows_and_get_obs(self, rows, job_id, url, fetch=False):
        # col index in each row, total 3 rows, read, write and delete
        rt_index = 3
        tp_index = 5
        bw_index = 6
        rd_rt, rd_tp, rd_bw, wr_rt, wr_tp, wr_bw = None, None, None, None, None, None
        try:
            retries = 1
            # if not in read_write_del stage while
            while len(rows) != 3 and retries < 999999:
                retries += 1
                print('NO COSBENCH DATA FOUND FOR: {} retrying {} ..'.format(job_id,
                                                                             retries))
                rows = self.parse_html_for_cosbench(url)
                time.sleep(1)
            else:  # if in read_write_del stage
                if retries < 999999:
                    print('COSBENCH DATA FOUND FOR: {} after {} attempts..'.format(job_id,
                                                                                   retries))
                    if fetch:  # only when there was ValueError
                        print('FETCHED AGAIN: {} time'.format(retries))
                        rows = self.parse_html_for_cosbench(url)
                    print(rows)

            # read data:
            read_row = rows[0]
            rd_rt = float(read_row[rt_index].strip().split()[0].strip())  # ms
            rd_tp = float(read_row[tp_index].strip().split()[0].strip())  # op/s
            rd_bw = read_row[bw_index]  # variable unit

            if rd_bw.strip().split()[1] == 'KB/S':
                rd_bw = float(rd_bw.strip().split()[0]) / 1000.0
            else:
                rd_bw = float(rd_bw.strip().split()[0])

            # write data:
            write_row = rows[1]
            wr_rt = float(write_row[rt_index].strip().split()[0].strip())  # ms
            wr_tp = float(write_row[tp_index].strip().split()[0].strip())  # op/s
            wr_bw = write_row[bw_index]
            if wr_bw.strip().split()[1] == 'KB/S':
                wr_bw = float(wr_bw.strip().split()[0]) / 1000.0
            else:
                wr_bw = float(wr_bw.strip().split()[0])

        except ValueError as e:
            print(e)
            print('Retrying Parsing...')
            rd_rt, rd_tp, rd_bw, wr_rt, wr_tp, wr_bw = self.check_get_rows_and_get_obs(rows,
                                                                         job_id,
                                                                         url,
                                                                         fetch=True)
        except TypeError as e:
            print(e)
            print('Retrying Parsing...')
            rd_rt, rd_tp, rd_bw, wr_rt, wr_tp, wr_bw = self.check_get_rows_and_get_obs(rows,
                                                                         job_id,
                                                                         url,
                                                                         fetch=True)
        except AttributeError as e:
            print(e)
            print('Retrying Parsing...')
            time.sleep(3)
            rd_rt, rd_tp, rd_bw, wr_rt, wr_tp, wr_bw = self.check_get_rows_and_get_obs(rows,
                                                                         job_id,
                                                                         url,
                                                                         fetch=True)

        finally:
            return rd_rt, rd_tp, rd_bw, wr_rt, wr_tp, wr_bw

    # REWARD
    def get_cosbench(self):
        """returns the cosbench scores of self.step_count timestep which will
        be used to calculate the reward
        :return: Response Time
        """
        # get latest job_id

        cmd = "pdsh -w cosbench 'cd {}; echo $(ls -dt w*/ | head -1)'".format(
            self.cosbench_log_folder)  # cosbench: w802-workmix2/
        # print (cmd)
        # w802-workmix2/
        p = Popen(cmd, shell=True, stdout=PIPE)
        var = p.stdout.read().strip()
        p.communicate()
        p.wait()
        # regular expression can be used
        print(var)
        var = str(var).strip().split()[1]
        print(var)
        job_id = var[var.index('w') + 1:var.index('-')]  # w802
        job_id = int(job_id) + 1
        job_id = 'w' + str(job_id)
        print(job_id)
        cosbench_url = self.cosbench_url
        url = cosbench_url + job_id
        print(url)

        rows = self.parse_html_for_cosbench(url)
        # get floats
        rd_rt, rd_tp, rd_bw, wr_rt, wr_tp, wr_bw = self.check_get_rows_and_get_obs(rows,
                                                                     job_id,
                                                                     url)
        # self.maxes [rhs, whs, rhl, whl]
        read_heavy = rd_tp > wr_tp
        if read_heavy:
            large_obj = rd_bw > rd_tp
            if large_obj:  # bw
                return self.get_slot_reward(rd_bw, self.maxes[2], self.n_slots)
            else:
                return self.get_slot_reward(rd_tp, self.maxes[0], self.n_slots)
        else:
            large_obj = wr_bw > wr_tp
            if large_obj:  # bw
                return self.get_slot_reward(wr_bw, self.maxes[3], self.n_slots)
            else:
                return self.get_slot_reward(wr_tp, self.maxes[1], self.n_slots)

    @staticmethod
    def get_slots(max_val, n_slots):
        discrete = [i for i in range(0, max_val + n_slots, max_val // n_slots)]
        slots = [(a, b) for a, b in zip(discrete, discrete[1:])]
        return slots

    def get_slot_reward(self, value, max_value, n_slots):
        slots = self.get_slots(max_value, n_slots)
        for r, slot in enumerate(slots):
            if slot[0] <= value < slot[1]:
                return r
        else:
            return n_slots

    def check_all_nodes_running(self):
        """to check the status of nodes at any given time,
        :return: True(all runningwith desired processes runnning) or False"""

        command = ['ceph', 'osd', 'tree']
        p = Popen(command, stdout=PIPE)
        text = p.stdout.read()
        p.communicate()
        p.wait()

        status_list = []
        # node_list = [n[-1] for n in self.sar_node_names]
        # d = dict.fromkeys(node_list, 0)
        n = 18
        i = 0

        while n < (20 + self.node_num * 10) and i < self.node_num:
            status_list.insert(i, text.split()[n])
            n = n + 10
            i = i + 1

        j = 0
        flag = 0
        while j < len(status_list):
            if status_list[j] == 'down':
                flag = flag + 1
            j = j + 1

        if flag == 1:
            # print ("cluster down")
            return False
        # print ("cluster working")
        return True

    # TODO: Based on SAR data(observations) might need previous cosbench data, will call two functions, done in agent
    def act(self, new_affinities):
        """change the affinities of nodes in the cluster to received input

        :param new_affinities: numpy array with new affinities,
                               a 1D array of shape(num_nodes, 1)
        :return: True(when done)
        """

        assert new_affinities.shape[0] == self.node_num
        for i in range(0, self.node_num):
            p = Popen('ceph osd primary-affinity %d %f' %
                      (i, float(new_affinities[i])), shell=True)
            p.communicate()
            p.wait()
        return True


if __name__ == '__main__':
        print('Testing...')
        # ce = ClusterEnvironment()
        # print(ce.get_affinities())
        # print (ce.get_sar_all_nodes())
        # print(ce.check_all_nodes_running())
        # affinity_array = np.array([0.9, 0.8, 0.7, 0.6 , 1.0, 1.0, 1.0, 1.0])
        # print(ce.act(affinity_array))
        # print(ce.get_cosbench())
        # ce.start_cluster()
        # ce.stop_cluster()
        print('Done...')