import pandas as pd
import matplotlib.pyplot as plt

# GLOBAL VARS
data = pd.read_csv("network_backup_dataset.csv")
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
work_flows = ['work_flow_0','work_flow_1','work_flow_2','work_flow_3','work_flow_4']

def extract_values(dict_flow, k):
    L = []
    for i in range(1,k+1):
        L.append(dict_flow[i])
    return L



def traverse_through_acc(flow, critical_day, no_of_weeks):
    strip_data = data[['Week #', 'Day of Week', 'Work-Flow-ID', 'Size of Backup (GB)']].values
    dict_flow = {}

    iterate_day = 1
    week_temp = 1
    day_temp = 'Monday'

    for a,b,c,d in strip_data:

        if no_of_weeks == int(a) and days[critical_day] == b:
            break

        if week_temp != a or day_temp != b:
            iterate_day = iterate_day + 1
            week_temp = a
            day_temp = b
        
        if c == flow:
            if iterate_day not in dict_flow:
                dict_flow[iterate_day] = d
            else:
                dict_flow[iterate_day] = dict_flow[iterate_day] + d

    flow = extract_values(dict_flow, iterate_day)
    return flow




def plot_relationship(k = 20):
    if k == 20:
        no_of_weeks = 3
        critical_day = 6
    elif k == 105:
        no_of_weeks = 16
        critical_day = 0
    else:
        print "Error - please enter a feasible number!"

    i = 0
    dict_flow = {}

    for flow in work_flows:
        dict_flow[i] = traverse_through_acc(flow, critical_day, no_of_weeks)
        i = i + 1
    return dict_flow





if __name__ == '__main__':
    #######################################################
    # Part a
    X = range(1,21)
    vec_20 = plot_relationship(20)
    for i in range(0, 5):
        plt.plot(X, vec_20[i], label = work_flows[i])


    plt.xlabel('Day Number', size=15)
    plt.ylabel('Backup Size', size=15)
    plt.title('Trends of Backup Size (20) w.r.t Flow', size=15)
    plt.legend(fontsize=7, loc = 'upper right')
    plt.draw()
    plt.savefig('D1_load_a.png', bbox_inches='tight')
    plt.show()

    #######################################################
    # Part b
    X = range(1,106)
    vec_105 = plot_relationship(105)
    for i in range(0, 5):
        plt.plot(X, vec_105[i], label = work_flows[i])


    plt.xlabel('Day Number', size=15)
    plt.ylabel('Backup Size', size=15)
    plt.title('Trends of Backup Size (105) w.r.t Flow', size=15)
    plt.legend(fontsize=7, loc = 'upper right')
    plt.draw()
    plt.savefig('D1_load_b.png', bbox_inches='tight')
    plt.show()

    #######################################################          
    # Part c
    '''    
    There are many trends that may be observed from the plots. It is clear 
    that workflows push periodically through the days. In specific,
    a. Workflow 4 backups more data near the weekend.
    b. Workflow 1 has the widest range in backup size across days.
    c. Workflow 2 has the lowest peak backup size.
    d. All backup trends are reasonably periodic.
    etc..
    '''





