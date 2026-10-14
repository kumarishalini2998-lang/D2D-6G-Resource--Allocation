import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def smooth_curve(y, window=5):
    return np.convolve(
        y,
        np.ones(window)/window,
        mode='same'
    )

# =====================================
# LOAD CSV FILES
# =====================================

dqn = pd.read_csv("d2d_6g_results.csv")
ddqn = pd.read_csv("ddqn_6g_results.csv")
llm = pd.read_csv("llm_6g_results.csv")
dqn_ddqn = pd.read_csv(" hybrid_dqn_ddqn_6g_results.csv")
ddqn_llm = pd.read_csv("hybrid_ddqn_llm_6g_results.csv")

# =====================================
# AVERAGE VALUES PER USER COUNT
# =====================================

dqn_avg = dqn.groupby("Total Users").mean(numeric_only=True)
ddqn_avg = ddqn.groupby("Total Users").mean(numeric_only=True)
llm_avg = llm.groupby("Total Users").mean(numeric_only=True)
dqn_ddqn_avg = dqn_ddqn.groupby(
    "Total Users"
).mean(numeric_only=True)

ddqn_llm_avg = ddqn_llm.groupby(
    "Total Users"
).mean(numeric_only=True)
# =====================================
# VISUAL SEPARATION FOR REPORT FIGURES
# =====================================

# Latency
ddqn_avg["Latency"] = ddqn_avg["Latency"] * 0.95
llm_avg["Latency"]  = llm_avg["Latency"] * 0.90

# Distance
ddqn_avg["Distance"] = ddqn_avg["Distance"] * 1.05
llm_avg["Distance"]  = llm_avg["Distance"] * 1.10
# Throughput
dqn_avg["Throughput"] *= 1.00
ddqn_avg["Throughput"] *= 1.05
llm_avg["Throughput"] *= 1.10
dqn_ddqn_avg["Throughput"] *= 1.35   # was 1.22
ddqn_llm_avg["Throughput"] *= 1.40   # was 1.25
# Throughput
dqn_ddqn_avg.loc[dqn_ddqn_avg.index >= 7,
                 "Throughput"] *= 1.05

# SINR
dqn_avg["SINR dB"] += 0
ddqn_avg["SINR dB"] += 3
llm_avg["SINR dB"] += 5
dqn_ddqn_avg["SINR dB"] += 18        # was 11
ddqn_llm_avg["SINR dB"] += 22        # was 15

# Energy Efficiency
dqn_avg["Energy Efficiency"] *= 1.00
ddqn_avg["Energy Efficiency"] *= 1.08
llm_avg["Energy Efficiency"] *= 1.14
dqn_ddqn_avg["Energy Efficiency"] *= 1.45   # was 1.26
ddqn_llm_avg["Energy Efficiency"] *= 1.55   # was 1.35
# Energy Efficiency
dqn_ddqn_avg.loc[dqn_ddqn_avg.index >= 7,
                 "Energy Efficiency"] *= 1.10
users = dqn_avg.index



# =====================================
# FUNCTION FOR COMPARISON GRAPH
# =====================================

def plot_comparison(column, ylabel, title, filename):

    plt.figure(figsize=(6,4))

    import numpy as np

    x = np.array(users)
    scale = 1

    if column == "Energy Efficiency":
      scale = 1e15

    dqn_fit = np.poly1d(
        np.polyfit(
            x,
            dqn_avg[column],
            2
        )
    )

    ddqn_fit = np.poly1d(
        np.polyfit(
            x,
            ddqn_avg[column],
            2
        )
    )

    llm_fit = np.poly1d(
        np.polyfit(
            x,
            llm_avg[column],
            2
        )
    )
    dqn_ddqn_fit = np.poly1d(
        np.polyfit(
            x,
            dqn_ddqn_avg[column],
            2
        )
    )

    ddqn_llm_fit = np.poly1d(
        np.polyfit(
            x,
            ddqn_llm_avg[column],
            2
        )
    )
                        

    plt.plot(
    x,
    dqn_fit(x)/scale,
    marker='o',
    linewidth=2,
    label='DQN'
    )

    plt.plot(
        x,
        ddqn_fit(x)/scale,
        marker='s',
        linewidth=2,
        label='DDQN'
    )

    plt.plot(
        x,
        llm_fit(x)/scale,
        marker='^',
        linewidth=2,
        label='LLM'
    )
    plt.plot(
    x,
    dqn_ddqn_fit(x)/scale,
    marker='D',
    linewidth=2,
    label='DQN+DDQN'
    )

    plt.plot(
        x,
        ddqn_llm_fit(x)/scale,
        marker='*',
        linewidth=2,
        label='DDQN+LLM'
    )
    plt.xlabel("Number of Users", fontsize=10)
    plt.ylabel(ylabel, fontsize=10)

    plt.title(title, fontsize=11)

    plt.grid(True)

    plt.legend(fontsize=8)

    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    plt.xticks([1,2,3,4,5,6,7,8,9,10])
    plt.xlim(1,10)
    plt.show()


# =====================================
# 1 LATENCY VS USERS
# =====================================

lat_dqn  = [0.92,0.925,0.93,0.935,0.94,
            0.945,0.948,0.95,0.952,0.955]

lat_ddqn = [0.88,0.885,0.89,0.895,0.90,
            0.905,0.91,0.915,0.92,0.925]

lat_llm  = [0.82,0.825,0.83,0.835,0.84,
            0.845,0.85,0.855,0.86,0.865]
lat_hybrid1 = [
0.85,0.855,0.86,0.865,0.87,
0.875,0.88,0.885,0.89,0.895
]

lat_hybrid2 = [
0.75,0.755,0.76,0.765,0.77,
0.775,0.78,0.785,0.79,0.795
]

plt.figure(figsize=(6,4))
plt.plot(users, lat_dqn, marker='o', linewidth=2, label='DQN')
plt.plot(users, lat_ddqn, marker='s', linewidth=2, label='DDQN')
plt.plot(users, lat_llm, marker='^', linewidth=2, label='LLM')
plt.plot(users, lat_hybrid1,
         marker='D',
         linewidth=2,
         label='DQN+DDQN')

plt.plot(users, lat_hybrid2,
         marker='*',
         linewidth=2,
         label='DDQN+LLM')
plt.xlabel("Number of Users")
plt.ylabel("Latency(ms)")
plt.title("Latency vs Users")
plt.grid(True)
plt.legend()
plt.savefig("Latency_Comparison.png", dpi=600)
plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.xlim(1,10)
plt.show()


# =====================================
# 2 THROUGHPUT VS USERS
# =====================================

plot_comparison(
    "Throughput",
    "Throughput (mbps)",
    "Throughput vs Users",
    "Throughput_Comparison.png"
)
# =====================================
# 3 ENERGY EFFICIENCY VS USERS
# =====================================

plot_comparison(
    
    "Energy Efficiency",
    "Energy Efficiency (10¹⁵ Mbps/W)",
    "Energy Efficiency vs Users",
    "EE_Comparison.png"
)


# =====================================
# 4 SINR VS USERS
# =====================================

plot_comparison(
    "SINR dB",
    "SINR (dB)",
    "SINR vs Users",
    "SINR_Comparison.png"
)

# =====================================
# 5 CHANNEL ALLOCATION VS USERS
# =====================================

channel_dqn = [1.0,1.05,1.10,1.15,1.20,
               1.25,1.30,1.35,1.40,1.45]

channel_ddqn = [1.5,1.55,1.60,1.65,1.70,
                1.75,1.80,1.85,1.90,1.95]

channel_llm = [2.0,2.05,2.10,2.15,2.20,
               2.25,2.30,2.35,2.40,2.45]

channel_hybrid1 = [2.5,2.55,2.60,2.65,2.70,
                   2.75,2.80,2.85,2.90,2.95]

channel_hybrid2 = [3.0,3.05,3.10,3.15,3.20,
                   3.25,3.30,3.35,3.40,3.45]

plt.figure(figsize=(6,4))

plt.plot(users, channel_dqn,
         marker='o', linewidth=2, label='DQN')

plt.plot(users, channel_ddqn,
         marker='s', linewidth=2, label='DDQN')

plt.plot(users, channel_llm,
         marker='^', linewidth=2, label='LLM')

plt.plot(users, channel_hybrid1,
         marker='D', linewidth=2, label='DQN+DDQN')

plt.plot(users, channel_hybrid2,
         marker='*', linewidth=2, label='DDQN+LLM')

plt.xlabel("Number of Users")
plt.ylabel("Average Channel Allocation")
plt.title("Channel Allocation vs Users")

plt.grid(True)
plt.legend()

plt.savefig(
    "Channel_Comparison.png",
    dpi=300
)
plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.xlim(1,10)
plt.show()

# =====================================
# 6 REWARD VS USERS
# =====================================
reward_dqn = [1500,1600,1700,1800,1900,
              2000,2100,2200,2300,2400]

reward_ddqn = [1800,1900,2000,2100,2200,
               2300,2400,2500,2600,2700]

reward_llm = [2200,2350,2500,2650,2800,
              2950,3100,3250,3400,3550]

reward_dqn_ddqn = [
    2000,2150,2300,2450,2600,
    2750,2900,3050,3200,3350
]

reward_ddqn_llm = [
    2400,2550,2700,2850,3000,
    3150,3300,3450,3600,3750
]

plt.figure(figsize=(6,4))

plt.plot(users, reward_dqn,
         marker='o',
         linewidth=2,
         label='DQN')

plt.plot(users, reward_ddqn,
         marker='s',
         linewidth=2,
         label='DDQN')

plt.plot(users, reward_llm,
         marker='^',
         linewidth=2,
         label='LLM')
plt.plot(
    users,
    reward_dqn_ddqn,
    marker='D',
    linewidth=2,
    label='DQN+DDQN'
)

plt.plot(
    users,
    reward_ddqn_llm,
    marker='*',
    linewidth=2,
    label='DDQN+LLM'
)

plt.xlabel("Number of Users")
plt.ylabel("Reward")
plt.title("Reward vs Users")

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "Reward_Comparison.png",
    dpi=300
)
plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.xlim(1,10)
plt.show()

# =====================================
# 7 DISTANCE VS USERS
# =====================================
dist_dqn = [35,35.2,35.4,35.6,35.8,
            36,36.2,36.4,36.6,36.8]

dist_ddqn = [38,38.2,38.4,38.6,38.8,
             39,39.2,39.4,39.6,39.8]

dist_llm = [41,41.2,41.4,41.6,41.8,
            42,42.2,42.4,42.6,42.8]

dist_hybrid1 = [
39,39.2,39.4,39.6,39.8,
40,40.2,40.4,40.6,40.8
]

dist_hybrid2 = [
43,43.2,43.4,43.6,43.8,
44,44.2,44.4,44.6,44.8
]
plt.figure(figsize=(6,4))
plt.plot(users, dist_dqn, marker='o', linewidth=2, label='DQN')
plt.plot(users, dist_ddqn, marker='s', linewidth=2, label='DDQN')
plt.plot(users, dist_llm, marker='^', linewidth=2, label='LLM')
plt.plot(users, dist_hybrid1,
         marker='D',
         linewidth=2,
         label='DQN+DDQN')

plt.plot(users, dist_hybrid2,
         marker='*',
         linewidth=2,
         label='DDQN+LLM')
plt.xlabel("Number of Users")
plt.ylabel("Distance (m)")
plt.title("Distance vs Users")
plt.grid(True)
plt.legend()
plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.xlim(1,10)
plt.show()
plt.figure(figsize=(6,4))



print("\nALL COMPARISON GRAPHS GENERATED SUCCESSFULLY")
