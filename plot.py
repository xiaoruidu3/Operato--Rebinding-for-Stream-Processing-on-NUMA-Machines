import matplotlib.pyplot as plt
import numpy as np
import json
import sys

config_path = './config.json'
out_file_template = "{workspace}/{exp_name}_{binary_name}_{threads}_{speed_up_ratio}_{task_size}_{policy}_{delay_tolerance}_{agent_number}_55_0_{out_or_err}.txt"

cms_keys_dic = {
    'citymos_const_multi_pri_0': 'Static operator binding',
    'citymos_const_multi_pri_1': 'Static partition binding',
    'citymos_const_multi_pri_2': 'Dynamic operator binding',
    'citymos_const_multi_pri_3': 'Dynamic partition binding',
    'citymos_const_multi_pri_4': 'Static metis binding',
    'citymos_const_multi_pri_5': 'Dynamic metis binding',
    'citymos_const_multi_pri_6': 'Metis with CriticalPath',
    'citymos_const_single_pri_100': 'Centralized task queue',
    'citymos_const_numa_pri_100': 'Per-NUMA task queue'
}

cms_vary_keys_dic = {
    'citymos_vary_multi_pri_0': 'Static operator binding',
    'citymos_vary_multi_pri_1': 'Static partition binding',
    'citymos_vary_multi_pri_2': 'Dynamic operator binding',
    'citymos_vary_multi_pri_3': 'Dynamic partition binding',
    'citymos_vary_multi_pri_4': 'Static metis binding',
    'citymos_vary_multi_pri_5': 'Dynamic metis binding',
    'citymos_vary_multi_pri_6': 'Metis with CriticalPath',
    'citymos_vary_single_pri_100': 'Centralized task queue',
    'citymos_vary_numa_pri_100': 'Per-NUMA task queue'
}

cms_const_keys_dic = {
    'citymos_const_const_multi_pri_0': 'Static operator binding',
    'citymos_const_const_multi_pri_1': 'Static partition binding',
    'citymos_const_const_multi_pri_2': 'Dynamic operator binding',
    'citymos_const_const_multi_pri_3': 'Dynamic partition binding',
    'citymos_const_const_multi_pri_4': 'Static metis binding',
    'citymos_const_const_multi_pri_5': 'Dynamic metis binding',
    'citymos_const_const_multi_pri_6': 'Metis with CriticalPath',
    'citymos_const_single_pri_100': 'Centralized task queue',
    'citymos_const_numa_pri_100': 'Per-NUMA task queue'
}

colors = {
    'Static operator binding': 'tab:blue',
    'Static partition binding': 'tab:purple',
    'Dynamic operator binding': 'tab:green',
    'Dynamic partition binding': 'tab:orange',
    'Static metis binding': 'tab:pink',
    'Dynamic metis binding': 'tab:olive',
    'Metis with CriticalPath': 'tab:cyan',
    'Centralized task queue': 'tab:red',
    'Per-NUMA task queue': 'tab:brown'
}


class Metric:
    def __init__(self):
        self.simTime = 0
        self.latency = 0


def setup_plt_parameters():
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['font.family'] = 'Linux Libertine O'
    # set title
    plt.rcParams['axes.titlesize'] = 50
    # set labels
    plt.rcParams['axes.labelsize'] = 50
    # set tickets
    plt.rcParams['xtick.labelsize'] = 40
    plt.rcParams['ytick.labelsize'] = 40
    # legend parameters
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['legend.loc'] = 'upper left'
    # grid parameter set up
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.grid'] = True


def legend_setup(fig, figxs, ax2):
    handles, labels = figxs.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    # labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    handles.append(handles_2[0])
    labels.append(labels_2[0])
    return fig.legend(handles, labels, bbox_to_anchor=(0, 1.08, 1, 0), fontsize=40, loc='center', borderaxespad=0,
                      ncol=3, frameon=False, columnspacing=1, markerscale=1)


def logbins_legend_setup(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    return fig.legend(handles, labels, bbox_to_anchor=(0, 1.08, 1, 0), fontsize=40, loc='center', borderaxespad=0,
                      ncol=3,
                      frameon=False, columnspacing=1, markerscale=1)


def parse_config():
    with open(config_path, mode='r') as f:
        data = json.load(f)
        workspace = data['workspace_path']

        return workspace


def __extract_avg_min_max_latencies(path: str):
    avg_delay = []
    max_delay = []
    min_delay = []
    timestamp = []

    with open(path, 'r') as file:
        lines = file.readlines()
        delay_line_id = 0
        for i in range(len(lines)):
            l = lines[i].strip()
            print(l)
            if (l == 'average latency avg-max-min'):
                delay_line_id = i + 1
                break

        delay_data = lines[delay_line_id].strip().split(',')
        delay_data.pop(-1)
        for d in delay_data:
            info = d.split(':')
            delays = info[1].split('-')
            timestamp.append(int(info[0]))
            avg_delay.append(float(delays[0]))
            max_delay.append(float(delays[1]))
            min_delay.append(float(delays[2]))

    return timestamp, avg_delay, max_delay, min_delay


def __extract_agent_num(path: str):
    with open(path, 'r') as file:
        lines = file.readlines()
        agent_number_line_id = 0
        for i in range(len(lines)):
            l = lines[i].strip()
            print(l)
            if (l == 'agent number'):
                agent_number_line_id = i + 1
                break

        agent_num = lines[agent_number_line_id].strip().split(',')
        agent_num.pop(-1)
        agent_num = [int(agent_num[i]) for i in range(len(agent_num))]

    return agent_num


def __plot_latency_lines(data, exp_name, thread_num, speed_up_ratio, task_size, delay_tolerance,
                         agent_number):
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    ax2 = ax.twinx()

    plot_data = {}
    plot_timestamp = []
    agent_size = []

    for binary_name in data[exp_name]['binary_name']:
        print(binary_name)
        if 'multi' in binary_name:
            for policy in data[exp_name]['policies']:
                key = binary_name + '_' + str(policy)
                partial_out_path = out_file_template.format(exp_name=exp_name, binary_name=binary_name,
                                                            threads=thread_num,
                                                            speed_up_ratio=speed_up_ratio,
                                                            task_size=task_size,
                                                            policy=policy, delay_tolerance=delay_tolerance,
                                                            agent_number=agent_number,
                                                            out_or_err="stdout")
                timestamp, avg_delay, max_delay, min_delay = __extract_avg_min_max_latencies(partial_out_path)
                plot_timestamp = timestamp
                plot_data[key] = avg_delay
                agent_size = __extract_agent_num(partial_out_path)
                if len(avg_delay) <= 0:
                    print('check log file: ', partial_out_path)
        else:
            key = binary_name + '_' + str(100)
            partial_out_path = out_file_template.format(exp_name=exp_name, binary_name=binary_name,
                                                        threads=thread_num,
                                                        speed_up_ratio=speed_up_ratio,
                                                        task_size=task_size,
                                                        policy=100, delay_tolerance=delay_tolerance,
                                                        agent_number=agent_number,
                                                        out_or_err="stdout")
            timestamp, avg_delay, max_delay, min_delay = __extract_avg_min_max_latencies(partial_out_path)
            plot_data[key] = avg_delay
            agent_size = __extract_agent_num(partial_out_path)
            if len(avg_delay) <= 0:
                print('check log file: ', partial_out_path)

    plot_timestamp = np.array(plot_timestamp) / speed_up_ratio
    

    for key, value in plot_data.items():
        if 'vary' in binary_name:
            ax.plot(plot_timestamp[:len(plot_timestamp) - 30], value[30:], linewidth=5, label=cms_keys_dic[key], color=colors[cms_keys_dic[key]])
        else:
            ax.plot(plot_timestamp, value, linewidth=5, label=cms_keys_dic[key], color=colors[cms_keys_dic[key]])

    ax2.plot(plot_timestamp, agent_size[:len(plot_timestamp)], linewidth=10, alpha=0.4, label='Vehicle Number',
             color='black')  # FIXME: put the right agent number in the second plot_timestamp.
    # ax2.set_ylim(0, 350000)

    ax.set_ylabel("Average latency [ms]")
    ax.set_xlabel("Wall clock time [s]")
    ax.set_yscale('symlog')
    ax.set_ylim((1e1, 10e4))
    if 'vary' in binary_name:
        ax.set_ylim((1e-1, 10e4))
    ax.set_xlim(plot_timestamp[0], 30)

    leg = legend_setup(fig, ax, ax2)
    plt.tight_layout()
    # plt.legend()
    # ax.legend()
    # ax2.legend('upper left')
    ax2.grid(False)

    plt.savefig(exp_name + '_' + str(thread_num) + '_' + str(speed_up_ratio) + '_' + str(task_size) + '_' + str(
        delay_tolerance) + '_' + str(agent_number) + '_'  + '_' + '_lines.png',
                bbox_extra_artists=[leg], bbox_inches='tight')

    # ax.cla()
    # ax2.cla()
    plt.close(fig)

def __extract_latency_counts(path: str):
    latency_counter = []
    with open(path, 'r') as file:
        lines = file.readlines()
        delay_line_id = 0
        for i in range(len(lines)):
            l = lines[i].strip()
            if (l == 'latency distribution'):
                delay_line_id = i + 1
                break

        delay_bins = lines[delay_line_id].strip().split(',')
        delay_bins.pop(-1)
        for d in delay_bins:
            latency_counter.append(int(d))

    return latency_counter


def __plot_latency_bins(data, exp_name, thread_num, speed_up_ratio, task_size, delay_tolerance,
                        agent_number):
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))

    plot_data = {}
    bin_num = 10
    bin_interval = 10

    for binary_name in data[exp_name]['binary_name']:
        print(binary_name)
        if 'multi' in binary_name:
            for policy in data[exp_name]['policies']:
                key = binary_name + '_' + str(policy)
                partial_out_path = out_file_template.format(exp_name=exp_name, binary_name=binary_name,
                                                            threads=thread_num,
                                                            speed_up_ratio=speed_up_ratio,
                                                            task_size=task_size,
                                                            policy=policy, delay_tolerance=delay_tolerance,
                                                            agent_number=agent_number,
                                                            out_or_err="stdout")
                latency_bins = __extract_latency_counts(partial_out_path)
                plot_data[key] = latency_bins
        else:
            key = binary_name + '_' + str(100)
            partial_out_path = out_file_template.format(exp_name=exp_name, binary_name=binary_name,
                                                        threads=thread_num,
                                                        speed_up_ratio=speed_up_ratio,
                                                        task_size=task_size,
                                                        policy=100, delay_tolerance=delay_tolerance,
                                                        agent_number=agent_number,out_or_err="stdout")
            latency_bins = __extract_latency_counts(partial_out_path)
            plot_data[key] = latency_bins

    x_ticket = ['~' + str(round(i * bin_interval, 1)) for i in range(1, bin_num + 1)]
    x_ticket[0] = '<{}'.format(round(bin_interval, 1))
    x_ticket[-1] = '>{}'.format(round((bin_num - 1) * bin_interval))

    x = np.arange(len(x_ticket))  # the label locations
    width = 0.1  # the width of the bars
    multiplier = 0
    interesting_latency_range = bin_interval * (bin_num - 1)

    for key, latency in plot_data.items():
        interesting_latencies = latency[:interesting_latency_range]
        non_interesting_latency = sum(latency[interesting_latency_range:])

        c = 1
        total_query_num = 0
        query_counter = [0 for i in range(bin_num)]
        for la in interesting_latencies:
            total_query_num += la
            if c % bin_interval == 0:
                query_counter[int(c / bin_interval) - 1] = total_query_num
                total_query_num = 0
            c += 1

        assert (query_counter[-1] == 0)
        query_counter[-1] = non_interesting_latency

        assert (sum(query_counter) == sum(latency))

        offset = width * multiplier
        ax.bar(x + offset, query_counter, width, label=cms_keys_dic[key], color=colors[cms_keys_dic[key]])
        multiplier += 1

    ax.set_xticks(x + 3.5 * width, x_ticket)
    ax.set_ylabel('Query count')
    ax.set_xlabel('Latency range [ms]')
    ax.set_yscale('log')
    ax.set_ylim((1e0, 10e5))
    ax.set_axisbelow(True)

    leg = logbins_legend_setup(fig, ax)
    plt.tight_layout()
    plt.savefig(exp_name + '_' + str(thread_num) + '_' + str(speed_up_ratio) + '_' + str(task_size) + '_' + str(
        delay_tolerance) + '_' + str(agent_number) + '_'  + '_bins.png',
                bbox_extra_artists=[leg], bbox_inches='tight')

    ax.cla()
    ax.cla()


def __extract_break_down(path: str):
    migration_cost = []
    compute_cost = []
    policy_cost = []

    with open(path, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            l = lines[i].strip()
            if l == 'migration_cost':
                migration_data = lines[i + 1].strip().split(',')
                migration_data.pop(-1)
                for d in migration_data:
                    migration_cost.append(int(d))

            if l == 'compute_cost':
                compute_data = lines[i + 1].strip().split(',')
                compute_data.pop(-1)
                for d in compute_data:
                    compute_cost.append(int(d))

            if l == 'policy_cost':
                policy_data = lines[i + 1].strip().split(',')
                policy_data.pop(-1)
                for d in policy_data:
                    policy_cost.append(int(d))

    return np.array(migration_cost[0:900], dtype=float), np.array(compute_cost[0:900], dtype=float), np.array(
        policy_cost[0:900], dtype=float)


def breakdown_legend_setup(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    return fig.legend(handles, labels, bbox_to_anchor=(0, 1.02, 1, 0), fontsize=40, loc='center', borderaxespad=0,
                      ncol=3,
                      frameon=False, columnspacing=1, markerscale=1)




def __plot_break_down(data, exp_name, thread_num, speed_up_ratio, task_size, delay_tolerance,
                      agent_number
                      ):
    h_bottom = 1
    h_top = 2

    # fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(20, 12),
                                            gridspec_kw={'height_ratios': [h_top, h_bottom]})
    for binary_name in data[exp_name]['binary_name']:
        print(binary_name)
        if 'multi' in binary_name:
            for policy in data[exp_name]['policies']:
                if policy != 3:
                    continue
                partial_out_path = out_file_template.format(exp_name=exp_name, binary_name=binary_name,
                                                            threads=thread_num,
                                                            speed_up_ratio=speed_up_ratio,
                                                            task_size=task_size,
                                                            policy=policy, delay_tolerance=delay_tolerance,
                                                            agent_number=agent_number,
                                                             out_or_err="stdout")
                migration_cost, compute_cost, policy_cost = __extract_break_down(partial_out_path)
                sum_val = np.copy(migration_cost)
                sum_val += compute_cost
                sum_val += policy_cost

                percentage_migration = migration_cost / sum_val * 100
                percentage_compute = compute_cost / sum_val * 100
                percentage_policy = policy_cost / sum_val * 100

                x = np.arange(len(compute_cost)) /  speed_up_ratio # x-axis: index / time step
                ax_bottom.stackplot(x,
                                    percentage_compute,
                                    percentage_migration,
                                    percentage_policy,
                                    labels=["Compute cost", "Migration cost", "Policy cost"],
                                    colors = ["darkgrey", "tab:orange", "tab:green"])

                ax_bottom.set_ylim(0, 99.7)

                ax_top.stackplot(x,
                                 percentage_compute,
                                 percentage_migration,
                                 percentage_policy,
                                 labels=["Compute cost", "Migration cost", "Policy cost"],
                                 colors = ["darkgrey", "tab:orange", "tab:green"])
                ax_top.set_ylim(99.7, 100)

                ax_bottom.set_xlim(0, 30)
                ax_top.set_xlim(0, 30)

                # ----- make the break look nice -----
                ax_bottom.spines['top'].set_visible(False)
                ax_top.spines['bottom'].set_visible(False)
                ax_top.tick_params(labelbottom=False)  # no x-tick labels on top

                # --- scale vertical part so both angles match visually ---
                d = 0.02  # size of diagonal lines

                # create different d for each axis
                dy_top = d
                dy_bottom = d * (h_top / h_bottom)

                kwargs_low = dict(transform=ax_bottom.transAxes, color='k', clip_on=False)
                kwargs_high = dict(transform=ax_top.transAxes, color='k', clip_on=False)

                # bottom of high axis
                ax_top.plot((-d, +d), (-dy_top, +dy_top), **kwargs_high)  # left
                ax_top.plot((1 - d, 1 + d), (-dy_top, +dy_top), **kwargs_high)  # right

                # top of low axis
                ax_bottom.plot((-d, +d), (1 - dy_bottom, 1 + dy_bottom), **kwargs_low)  # left
                ax_bottom.plot((1 - d, 1 + d), (1 - dy_bottom, 1 + dy_bottom), **kwargs_low)  # right


                xlab = fig.supxlabel("Wall clock time [s]", fontsize=40)
                ylab = fig.supylabel("Cost Percentage [%]", fontsize=40)

                leg = breakdown_legend_setup(fig, ax_bottom)
                plt.tight_layout()

                plt.savefig(
                    exp_name + '_' + str(thread_num) + '_' + str(speed_up_ratio) + '_' + str(task_size) + '_' + str(
                        delay_tolerance) + '_' + str(agent_number) + '_' + '_breakdown.png', bbox_extra_artists=[leg, xlab, ylab], bbox_inches='tight')

                ax_top.cla()
                ax_top.cla()

                ax_bottom.cla()
                ax_bottom.cla()


def plot(flag):
    global out_file_template, cms_keys_dic, colors, cms_vary_keys_dic
    workspace = parse_config()
    out_file_template = out_file_template.format(workspace=workspace, exp_name='{exp_name}',
                                                 binary_name='{binary_name}', threads='{threads}',
                                                 speed_up_ratio='{speed_up_ratio}', task_size='{task_size}',
                                                 policy='{policy}',
                                                 delay_tolerance='{delay_tolerance}', agent_number='{agent_number}',
                                                out_or_err='{out_or_err}')

    setup_plt_parameters()
    with open(config_path, mode='r') as f:
        data = json.load(f)
        exp_names = data['exp_names']
        for exp_name in exp_names:
            if 'vary' in exp_name:
                cms_keys_dic = cms_vary_keys_dic
            if 'const_const' in exp_name:
                cms_keys_dic = cms_const_keys_dic
            exp_info = data[exp_name]
            if exp_info['enable']:
                for thread_num in data['threads']:
                    for task_size in exp_info['task_size']:
                        for speed_up_ratio in exp_info['speed_up_ratio']:
                            for delay_tolerance in exp_info['delay_tolerance']:
                                for agent_number in exp_info['agent_number']:
                                    if flag == 0:
                                        # ax2 = ax.twinx()
                                        __plot_latency_lines(data, exp_name, thread_num - 2,
                                                             speed_up_ratio,
                                                             task_size, delay_tolerance, agent_number)
                                    elif flag == 1:
                                        __plot_latency_bins(data, exp_name, thread_num - 2,
                                                            speed_up_ratio,
                                                            task_size,
                                                            delay_tolerance, agent_number)

                                    elif flag == 2:
                                        __plot_break_down(data, exp_name, thread_num - 2,
                                                          speed_up_ratio,
                                                          task_size,
                                                          delay_tolerance, agent_number)


if __name__ == '__main__':
    p = sys.argv[1]
    print(p)
    if sys.argv[1] == '--delay-lines':
        plot(0)
    elif sys.argv[1] == '--delay-bins':
        plot(1)
    elif sys.argv[1] == '--break-down':
        plot(2)
    else:
        print("invalid plot parameter")
