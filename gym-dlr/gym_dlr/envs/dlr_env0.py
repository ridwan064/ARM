# gym req
import gym
from gym import error, spaces, utils
from gym.utils import seeding

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

# SAR_NODE_NAMES = ['node1', 'node2', 'node4', 'node5', 'node6', 'node7', 'node8', 'node9']
SAR_NODE_NAMES = ['ceph-2', 'ceph-3', 'ceph-4', 'ceph-5', 'ceph-6', 'ceph-7', 'ceph-8']

MAXES = [2200, 1800, 320, 210]  # rhs, whs, rhl, whl


ALL_NODE_NAMES = ['ceph-1'] + SAR_NODE_NAMES
NUM_NODES = len(SAR_NODE_NAMES)  # cluster size excluding master node
SAR_LOG_DIR = '/home/ceph-user/sar-logs'
COSBENCH_LOGS_DIR = '/home/ceph-user/cos/archive'
PHANTOMJS_PATH = '/usr/local/share/phantomjs-2.1.1-linux-x86_64/bin/phantomjs'  #  <-------------
# PHANTOMJS_PATH = '/usr/local/share/phantomjs-1.9.8-linux-x86_64/bin/phantomjs'  # <-------------
COSBENCH_WEBGUI_URL_BASE = 'http://129.114.108.113:19088/controller/workload.html?id='
TOTAL_NET = 1310720  # KB, assuming maximum network bandwidth = 3.5Gbps = 458752KBps or 19200 KB if we assume max network bandwidth = 150Mbps (50 threads in iperf)

TOTAL_STATS = 7  # [io_wait_array, net_usage_array, rtps_array, wtps_array, cpu_usage_array, affinity_array, reweight_array]
USEFUL_STATS = 4  # [io_wait_array, net_usage_array, rtps_array, wtps_array] # <---should be in same order

WAIT = 20  # in seconds
N_SLOTS = 20

RETRY = 30
"""-------2. INTERFERENCE FUNCTIONS-------"""


# TODO: randomize with 2(one of the intereferece) and 2(another one) sysbench nodes

def execute_cpu(node_names_list):

    node_names = ",".join(node_names_list)
    cpu_interference = "pdsh -w {} 'sysbench --num-threads=512 --test=cpu --cpu-max-prime=50000000 run --max-time=3600s'".format(
        node_names)
    cpu_interference = "({};{})&".format(cpu_interference, cpu_interference)

    print('Running sysbench with cmd: {}'.format(cpu_interference))
    p3 = Popen(cpu_interference, shell=True)
    p3.communicate()
    p3.wait()


def execute_io(node_names_list):

    node_names = ",".join(node_names_list)
    io_interference = "pdsh -w {} 'cd /home/ceph-user/fio-data;fio --name=randread --ioengine=libaio --iodepth=1 --rw=randread --bs=1k --direct=0 --size=5G --numjobs=10 --runtime=3600'".format(
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
        self.interferences = [execute_net, execute_io]  # execute_net]
        # seq is important
        self.train_workloads = ["workload3.xml", "workload4.xml", "workload5.xml",  "workload6.xml", "workload1.xml", "workload2.xml"]

        self.large = ["workload3.xml", "workload4.xml", "workload6.xml"]
        self.write = ["workload2.xml", "workload4.xml"]
        self.balanced = ["workload5.xml",  "workload6.xml"]

        self.selected_workload = None
        self.bad_nodes = None
        self.selected_interference = None
        self.idx_pair = (0, 0)
        self.selected_mean = 0

        self.step_count = 0

        self.episode_number = 1
        self.means = [  0.1262,
                        0.3104,
                        0.3104,
                        0.1262,
                        0.0412,
                        0.019,
                        0.019,
                        0.0412,
                        0.6242,
                        0.4811,
                        0.4811,
                        0.6242,
                        0.0704,
                        0.0309,
                        0.0309,
                        0.0704,
                        5.7719,
                        5.3847,
                        5.3847,
                        5.7719,
                        0.3745,
                        0.1682,
                        0.1682,
                        2.5301
                    ]
        self.baseline = False
        self.raw_reward = False
        # self.start_cluster()

    def start_cluster(self, workload, interference, hm_bad_nodes):
        """
        THIS IS THE ENTRY POINT FOR ALL BENCHMARKS RUN: all setup processes
        for cluster done here
        :return: True when ready for agent to start working
        """

        print("Starting cluster...WORKLOAD: ", workload)
        self.selected_workload = workload
        random_interference = [interference]

        timestamp = '-'.join(datetime.today().__str__().split())[:-7]
        extnsn = 'cos_h'
        new_filename = '{}.{}'.format(timestamp, extnsn)

        # OBSERVATIONS: SAR
        sar_node_names = ",".join(self.sar_node_names)
        sar_cmd = "pdsh -w {} 'sar -b -u -r -n DEV 20 120' > {}/{} &".format(sar_node_names, self.sar_log_dir, new_filename) # update 11/30

        print('Running sar with cmd: {}'.format(sar_cmd))
        p1 = Popen(sar_cmd, shell=True)
        p1.communicate()
        p1.wait()
        print('Sleeping for 40s..Else no obs data..ERROR in agent: SHAPE')
        time.sleep(40)

        # REWARDS: COSBENCH
        cosbench_cmd = "cd /home/ceph-user/cos; sh cli.sh submit conf/{} &".format(workload)
        print('Running cosbench with cmd: {}'.format(cosbench_cmd))
        p2 = Popen(cosbench_cmd, shell=True)
        p2.communicate()
        p2.wait()

        if self.selected_workload in self.large:
            print("====> Sleeping for LARGE object prepare stage 60s..")
            time.sleep(120)

        else:
            print("====> Sleeping for SMALL object prepare stage 30s..")
            time.sleep(60)

        # RANDOM INTERFERENCES
        # size = np.random.choice([2, 4], replace=False)
        size = hm_bad_nodes
        random_nodes = list(np.random.choice(self.sar_node_names, size=size, replace=False))
        self.bad_nodes = random_nodes

        # random_interference = [np.random.choice(self.interferences, replace=False)]

        self.selected_interference = random_interference[0].__name__

        random_interference[0](random_nodes)
        # random_interference[0](random_nodes[:2])
        # random_interference[1](random_nodes[2:])

        return workload, random_nodes, [r.__name__ for r in random_interference]

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
        cosbench_info_command = "sh /home/ceph-user/cos/cli.sh info"
        p3 = Popen(cosbench_info_command, stdout=PIPE, shell=True)
        text = p3.stdout.read().strip()
        text = str(text).strip()
        p3.communicate()
        p3.wait()
        # print (text)
        # job_id = text.splitlines()[5].strip().split(":")[1].strip().split()[0].strip()
        wexp = r'(w[0-9]+)'
        finds = re.findall(wexp, text)
        found_job = len(finds) != 0
        if found_job:
            print("jobs found ", len(finds))
            for job_id in finds:
                cosbench_kill_cmd = "sh /home/ceph-user/cos/cli.sh cancel {}".format(
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

    @staticmethod
    def parse_data_regular_exp(data, expression):
        finds = re.findall(expression, data)
        parsed_data = []
        for f in finds:
            node_number = 0
            node_stats = []
            for d in f:
                if d.startswith('ceph-'):
                    node_number = d[-1]
                elif d != ' ':
                    node_stats.append(d)
            parsed_data.append((node_number, node_stats))

        parsed_data = sorted(parsed_data, key=lambda t: t[0])
        return parsed_data

    def parse_cpu_io(self, data, expression):
        parsed_data = self.parse_data_regular_exp(data, expression)
        cpu_usage_list = []
        io_wait_list = []

        for k, v in parsed_data:
            cpu1 = float(v[0])
            cpu2 = float(v[2])
            io = float(v[3])

            cpu_usage_list.append((cpu1 + cpu2) / 100)
            io_wait_list.append(io / 100)

        cpu_usage_array = np.array(cpu_usage_list)
        io_wait_array = np.array(io_wait_list)

        return cpu_usage_array, io_wait_array

    def parse_network(self, data, expression):
        parsed_data = self.parse_data_regular_exp(data, expression)
        net_usage_list = []
        total_net = self.total_net
        for k, v in parsed_data:
            rxkb = float(v[2])
            txkb = float(v[3])
            net = rxkb / total_net if rxkb > txkb else txkb / total_net
            net_usage_list.append(net)

        net_usage_array = np.array(net_usage_list)
        return net_usage_array

    def parse_read_write(self, data, expression):
        parsed_data = self.parse_data_regular_exp(data, expression)
        rtps_list = []
        wtps_list = []

        for k, v in parsed_data:
            rtps = float(v[1])
            wtps = float(v[2])

            # TODO
            rtps_list.append(rtps / 1000)   # normalization factor divide by experimental max
            wtps_list.append(wtps / 1000)

        rtps_array = np.array(rtps_list)
        wtps_array = np.array(wtps_list)

        return rtps_array, wtps_array

    # TO GET OBSERVATIONS
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
        raw_data = Popen(['tail', '-n', '119', latest_file], shell=False, stdout=PIPE) # update 11/30
        data = raw_data.stdout.read().strip()
        data = str(data)

        cpu_io_exp = r'(ceph-[1-9]):( )+[0-9][0-9]:[0-9][0-9]:[0-9][0-9]( )+[AP][M]( )+[a][l][l]( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)'
        net_exp = r'(ceph-[1-9]):( )+[0-9][0-9]:[0-9][0-9]:[0-9][0-9]( )+[AP][M]( )+[e][t][h][0]( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)'
        rw_exp = r'(ceph-[1-9]):( )+[0-9][0-9]:[0-9][0-9]:[0-9][0-9]( )+[AP][M]( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)( )+(\d+\.\d+)'
        cpu_usage_array, io_wait_array = self.parse_cpu_io(data, cpu_io_exp)
        net_usage_array = self.parse_network(data, net_exp)
        rtps_array, wtps_array = self.parse_read_write(data, rw_exp)

        # (7, 7)
        obs = np.array([io_wait_array, net_usage_array, rtps_array, wtps_array, cpu_usage_array, affinity_array, weight_array])  # affinity, weight is just there, will not be used
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

        print('Total rows as (r, w, d), Number of rows found out of 3: {}/3'.format(len(rows)))
        print('Cosbench Data: ', rows)

        return rows

    def check_get_rows_and_get_obs(self, rows, job_id, url, fetch=False, fetch_count=0):
        # col index in each row, total 3 rows, read, write and delete
        rt_index = 3
        tp_index = 5
        bw_index = 6
        rd_rt, rd_tp, rd_bw, wr_rt, wr_tp, wr_bw = None, None, None, None, None, None
        try:
            retries = 1
            # if not in read_write_del stage while
            fetch_unbalanced = len(rows) != 3 and self.selected_workload not in ["workload5.xml",
                                                                                 "workload6.xml"] and retries < RETRY
            fetch_balanced = self.selected_workload in ["workload5.xml",
                                                        "workload6.xml"] and len(rows) != 2 and retries < RETRY
            while fetch_unbalanced or fetch_balanced:
                retries += 1
                print('NO COSBENCH DATA FOUND FOR: {} retrying:{} fetch:{}..'.format(job_id,
                                                                                     retries, fetch))
                time.sleep(2)
                if self.selected_workload in self.large:
                    print('WORKLOAD: {}, PREPARE SLEEP: 60s'.format(self.selected_workload))
                    time.sleep(60)

                rows = self.parse_html_for_cosbench(url)
                fetch_unbalanced = len(rows) != 3 and self.selected_workload not in ["workload5.xml",
                                                                                     "workload6.xml"] and retries < RETRY
                fetch_balanced = self.selected_workload in ["workload5.xml",
                                                            "workload6.xml"] and len(rows) != 2 and retries < RETRY
            else:  # if in read_write_del stage
                if retries < 120:
                    print('COSBENCH DATA FOUND FOR: {} after {} attempts..'.format(job_id,
                                                                                   retries))
                    if fetch:  # only when there was ValueError (N/A)
                        time.sleep(1)
                        rows = self.parse_html_for_cosbench(url)
                        print('FETCHED AGAIN: {} time'.format(retries))
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
            print('Got N/A Retrying Parsing...')
            fc = fetch_count + 1
            if fc > 1:
                return None, None, None, None, None, None
            rd_rt, rd_tp, rd_bw, wr_rt, wr_tp, wr_bw = self.check_get_rows_and_get_obs((),
                                                                                       job_id,
                                                                                       url,
                                                                                       fetch=True,
                                                                                       fetch_count=fc)
        except TypeError as e:
            print(e)
            print('Got N/A Retrying Parsing...')
            fc = fetch_count + 1
            if fc > 1:
                return None, None, None, None, None, None
            rd_rt, rd_tp, rd_bw, wr_rt, wr_tp, wr_bw = self.check_get_rows_and_get_obs((),
                                                                                       job_id,
                                                                                       url,
                                                                                       fetch=True,
                                                                                       fetch_count=fc)
        except AttributeError as e:
            print(e)
            print('Retrying Parsing...')
            time.sleep(3)
            rd_rt, rd_tp, rd_bw, wr_rt, wr_tp, wr_bw = self.check_get_rows_and_get_obs((),
                                                                         job_id,
                                                                         url,
                                                                         fetch=True)

        finally:
            return rd_rt, rd_tp, rd_bw, wr_rt, wr_tp, wr_bw

    # TO GET REWARD
    def get_cosbench(self):
        """returns the cosbench scores of self.step_count timestep which will
        be used to calculate the reward
        :return: Response Time
        """
        # get latest job_id

        cmd = "cd {}; echo $(ls -dt w*/ | head -1)".format(
            self.cosbench_log_folder)  # cosbench: w802-workmix2/
        # print (cmd)
        # w802-workmix2/
        p = Popen(cmd, shell=True, stdout=PIPE)
        var = p.stdout.read().strip()
        p.communicate()
        p.wait()
        # regular expression can be used
        print(var)
        # var = str(var).strip().split()[1]
        # print(var)
        var = str(var)
        job_id = var.strip()[var.index('w') + 1:var.index('-')]  # w802
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

        # TODO: UPDATE MEANS OF EVERY CASE, FORMULA MAY CHANGE
        mean_idx = self.episode_number % len(self.means)
        self.selected_mean = self.means[mean_idx] / 1000

        if None in [rd_rt, rd_tp, rd_bw, wr_rt, wr_tp, wr_bw]:
            return -1, np.asarray([rd_rt, rd_tp, rd_bw, wr_rt, wr_tp, wr_bw])

        else:

            if self.selected_workload in self.balanced:
                reward = 2 / (rd_rt + wr_rt)

            elif self.selected_workload in self.write:
                reward = 1 / wr_rt

            else:
                reward = 1 / rd_rt

            print('@' * 60)
            print("REWARD: {}, TARGET: {}".format(reward, self.selected_mean))
            print('@' * 60)

            if not self.raw_reward:
                reward -= self.selected_mean
                if self.selected_interference == "execute_io":
                    gain = np.random.choice([1, 2], size=1)[0]
                    reward = gain * reward

            # self.maxes [rhs, whs, rhl, whl]
            # read_heavy = rd_tp > wr_tp

            # if self.selected_workload == "workload6.xml":
            #     reward = 2 / (rd_rt + wr_rt)
            #
            # elif self.selected_workload == "workload5.xml":
            #     reward = 2 / (rd_rt + wr_rt)
            #
            # else:
            #
            #     if read_heavy:
            #         large_obj = rd_bw > rd_tp
            #         if large_obj:  # bw
            #             reward = 1 / rd_rt
            #             # reward = self.get_slot_reward(rd_bw, self.maxes[2], self.n_slots)
            #         else:
            #             reward = 1 / rd_rt
            #             # reward = self.get_slot_reward(rd_tp, self.maxes[0], self.n_slots)
            #     else:
            #         large_obj = wr_bw > wr_tp
            #         if large_obj:  # bw
            #             reward = 1 / wr_rt
            #             # reward = self.get_slot_reward(wr_bw, self.maxes[3], self.n_slots)
            #         else:
            #             reward = 1 / wr_rt
            #             # reward = self.get_slot_reward(wr_tp, self.maxes[1], self.n_slots)

            return reward, np.asarray([rd_rt, rd_tp, rd_bw, wr_rt, wr_tp, wr_bw])

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
            return n_slots  # all values greater than max will be scored same / noise

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
    def change(self, new_values, change_affinity=1., baseline=False):
        """change the affinities of nodes in the cluster to received input

        :param new_values: numpy array with new affinities/weights,
                               a 1D array of shape(num_nodes, 1)
        :param change_affinity: Boolean
        :param baseline: Boolean take no action
        :return: True(when done)
        """
        if baseline:
            print('@' * 60)
            time.sleep(15)
            print("======> NO ACTION <======")
            print('@' * 60)
            return True

        if change_affinity == 2.:
            print('@' * 60)
            print("======> NO ACTION <======")
            print('@' * 60)
            return True
        assert new_values.shape[0] == self.node_num

        for node_num, value in enumerate(new_values):
            key = 'primary-affinity' if change_affinity == 1. else 'reweight'
            p = Popen('ceph osd {} {} {}'.format(key, node_num, float(value)), shell=True)
            p.communicate()
            p.wait()
        return True


"""-------4. GYM ENV-------"""


class DlrEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # will start the required processes in the cluster
        self.env = ClusterEnvironment()

        # actions: new values affinities/weights
        # (n_nodes + 1, 1) +1 flag for affinity or weight
        self.action_space = spaces.Box(low=0,
                                       high=1,
                                       shape=(NUM_NODES + 1, 1))

        # (n_params, n_nodes) = (7, 8) First 5 are stats, last is affinity & weight
        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=(TOTAL_STATS, NUM_NODES))

        self.status = self.env.check_all_nodes_running()

        self.batch_size = 0
        self.episode_number = 0
        self.train_workloads = self.env.train_workloads
        self.workload = np.random.choice(self.train_workloads, replace=False)
        self.bad_nodes = None
        self.interference = None
        self.test = False
        self.test_count = 0
        self.baseline = False
        self.hm_bad_nodes = 2
        self.idx_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.reward_target = None
        self.raw_reward = False

    def step(self, action):
        """Change affinities, set them in cluster, wait for observation 21s,
        return (ob, reward, episode_over, empyt_dict)"""

        # take action = set new values
        new_values = action.reshape(-1)
        assert new_values.shape[0] == NUM_NODES + 1
        change_affinity = new_values[-1]
        assert change_affinity in [0., 1., 2.]
        new_values = new_values[:NUM_NODES]
        assert new_values.shape[0] == NUM_NODES

        self.env.change(new_values, change_affinity, baseline=self.baseline)
        self.env.step_count += 1  # <----------------------

        print('@' * 60)
        print('Sleeping for 21s..inside step..before collecting stats after action..')
        time.sleep(21)
        print('@' * 60)

        # check the status and observe the status of nodes
        self.status = self.env.check_all_nodes_running()
        assert self.status, "Interaction pipeline says node status of cluster is not Good. Check it."

        # OBSERVATION FOR AGENT: (n_stats, n_nodes)
        ob = self.env.get_sar_all_nodes()

        # simultaneously get the performance metrics
        try:
            reward, cos_values = self.env.get_cosbench()
        except IndexError as e:
            print('-' * 60)
            print("Error: IndexError Sleep for 3s for prepare..", e)
            time.sleep(3)
            print('-' * 60)
            reward, cos_values = self.env.get_cosbench()
        except TypeError as e:
            print('-' * 60)
            print("Error: TypeError Sleep for 3s for prepare..", e)
            time.sleep(3)
            print('-' * 60)
            reward, cos_values = self.env.get_cosbench()
        except Exception as e:
            print('Error: ', e)
            print('-' * 60)
            print("Sleep for 3s for prepare..")
            time.sleep(3)
            print('-' * 60)
            reward, cos_values = self.env.get_cosbench()

        self.reward_target = self.env.selected_mean * 1000
        done = self.env.step_count == 10  # when step_count == 10 done <----------------------
        return ob, reward, done, {'COS_VALUES': cos_values,  # list
                                  'STEP': self.env.step_count}

    def reset(self):
        print('@' * 60)
        print('Stopping Cluster..Inside reset..')
        self.env.stop_cluster()

        print('Sleeping for 90s..inside reset..after 1st stop cluster')
        time.sleep(90)

        print('Will start cluster inside reset..')
        print('@' * 60)
        self.env = ClusterEnvironment()
        self.env.baseline = self.baseline
        self.env.raw_reward = self.raw_reward
        self.env.episode_number = self.episode_number - 1
        # self.idx_pairs = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 0), (1, 0), (0, 1), (1, 1)]

        if ((self.episode_number - 1) % self.batch_size == 0) and self.episode_number != 1:
            self.workload = np.random.choice(self.train_workloads, replace=False)

        if self.test:
            self.workload = self.train_workloads[self.test_count]
            if self.episode_number % len(self.idx_pairs) == 0:  # after 4 episodes use next workload
                self.test_count = (self.test_count + 1) % len(self.train_workloads)

        self.env.idx_pair = self.idx_pairs[self.episode_number % len(self.idx_pairs)]
        alt_interf_idx,  hm_bad_nodes_idx = self.env.idx_pair

        selected_interf = self.env.interferences[alt_interf_idx]
        self.hm_bad_nodes = [2, 2][hm_bad_nodes_idx]

        properties = self.env.start_cluster(self.workload, selected_interf, self.hm_bad_nodes)
        self.bad_nodes = self.env.bad_nodes
        self.interference = self.env.selected_interference

        print('@' * 60)
        print('WORKLOADS: ', properties[0])
        print('NODES: ', properties[1])
        print('INTERFERENCES: ', properties[2])
        print('@' * 60)

        print('Reset aff. and weight values to 1. and collect obs....')
        self.env.change(np.ones(shape=(NUM_NODES,)), change_affinity=True)
        self.env.change(np.ones(shape=(NUM_NODES,)), change_affinity=False)
        obs = self.env.get_sar_all_nodes()

        while len(obs.shape) != 2:
            time.sleep(1)
            print('ERROR: Collecting sar again because observation shape: ', obs.shape)
            obs = self.env.get_sar_all_nodes()
        return obs

    def render(self, mode='human', close=False):
        pass


if __name__ == '__main__':
    print('Testing...')
    # ce = ClusterEnvironment()
    # print(ce.get_affinities())
    # print (ce.get_sar_all_nodes())
    # print(ce.check_all_nodes_running())
    # affinity_array = np.array([0.9, 0.8, 0.7, 0.6 , 1.0, 1.0, 1.0, 1.0])
    # print(ce.act(affinity_array))
    # prin