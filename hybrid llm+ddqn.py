import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import random
import re
from transformers import pipeline
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    framework="pt"
)

 
from collections import deque

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model



PACKET_LOSS_PROB = 1e-7
TASK_SIZE_MB = 20
# =====================================
# DQN PARAMETERS
# =====================================

STATE_SIZE = 3
# Available spectrum channels (resource blocks)

CHANNELS = [0, 1, 2, 3]

ACTION_SIZE = len(CHANNELS)

GAMMA = 0.95

EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

LEARNING_RATE = 0.001

BATCH_SIZE = 32

memory = deque(maxlen=10000)
# Build the DQN Network
model = Sequential([
    Dense(128,
          input_dim=STATE_SIZE,
          activation='relu'),

    Dense(128,
          activation='relu'),

    Dense(ACTION_SIZE,
          activation='linear')
    ])


from tensorflow.keras.losses import Huber

model.compile(
         optimizer=Adam(
          learning_rate=LEARNING_RATE
    ),
    loss=Huber()
    )
target_model = clone_model(model)

target_model.set_weights(
    model.get_weights()
)
    
#Add  Action Selection Function
def choose_action(state):

    global EPSILON

    if np.random.rand() < EPSILON:

        return np.random.randint(
            ACTION_SIZE
        )

    q_values = model.predict(
        state.reshape(1,-1),
        verbose=0
    )

    return np.argmax(q_values[0])
# Add Replay Training Function
def replay():

    global EPSILON

    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(
        memory,
        BATCH_SIZE
    )

    for state, action, reward, next_state, done in batch:

        if done:
            target = reward

        else:

            next_action = np.argmax(
                model.predict(
                    next_state.reshape(1,-1),
                    verbose=0
                )[0]
            )

            next_q = target_model.predict(
                next_state.reshape(1,-1),
                verbose=0
            )[0]

            target = reward + GAMMA * next_q[next_action]

        target_f = model.predict(
            state.reshape(1,-1),
            verbose=0
        )

        target_f[0][action] = target

        model.fit(
            state.reshape(1,-1),
            target_f,
            epochs=1,
            verbose=0
        )

    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY
# =====================================================
# 6G D2D RESOURCE ALLOCATION SIMULATOR
# =====================================================

# -------- PARAMETERS --------

FC = 10e12                 # 10 THz Carrier Frequency
BANDWIDTH = 100e6          # 100 MHz

CELL_RADIUS = 500          # m
NUM_AI_DEVICES = 50

PATHLOSS_EXPONENT = 3

TX_POWER_DBM = 32
TX_POWER = 10 ** ((TX_POWER_DBM - 30) / 10)

D2D_MAX_DISTANCE = 30      # m

PACKET_SIZE_BITS = 20 * 8

PROCESSING_SPEED_MIN = 6
PROCESSING_SPEED_MAX = 52

MOBILITY_MIN = 1
MOBILITY_MAX = 2

REQUEST_PROBABILITY = 0.5

kB = 1.38e-23
TEMP = 290

NOISE_POWER = kB * TEMP * BANDWIDTH

SINR_THRESHOLD_DB = 25

FRAME_DURATION = 0.5e-3

PACKET_LOSS_PROB = 1e-7

TASK_SIZE_MB = 20

# =====================================================
# STORAGE
# =====================================================

all_results = []

user_counts = []
latency_results = []
throughput_results = []
ee_results = []
sinr_results = []
channel_results = []
reward_results = []
distance_results=[]
episode_rewards=[]



# =====================================================
# SIMULATION
# =====================================================
def llm_allocate_channel(
        distance,
        active_users,
        mobility,
        channel_gain,
        channel_interference):

    prompt = f"""
You are a 6G spectrum allocation agent.

Distance = {distance:.2f}
Active Users = {active_users}
Mobility = {mobility:.2f}

Interference values:

Channel0 = {channel_interference[0]:.8e}
Channel1 = {channel_interference[1]:.8e}
Channel2 = {channel_interference[2]:.8e}
Channel3 = {channel_interference[3]:.8e}

Choose the channel with the LOWEST interference.

Answer only:
0 or 1 or 2 or 3
"""


    response = llm(prompt,max_new_tokens=10)

    text = response[0]["generated_text"]

    print("LLM Decision:", text)

# LLM suggestion
    numbers = re.findall(r'\d+', text)

    if numbers:
       llm_channel = int(numbers[0]) % 4
    else:
       llm_channel = np.random.randint(0,4)

    scores = []

    for ch in range(4):
       score = 1 / (channel_interference[ch] + 1e-12)
       scores.append(score)

    best_channel = np.argmax(scores)

    print("Best Channel =", best_channel)

# 80% use best engineering choice
# 20% follow LLM suggestion

    if np.random.rand() < 0.8:
       channel = best_channel
    else:
       channel = llm_channel

    return channel
EPISODES = 50

for episode in range(EPISODES):
    episode_reward=0
    if episode % 5 == 0:
      target_model.set_weights(
        model.get_weights()
     )
      print("Target Network Updated")

    print(f"\nEpisode {episode+1}/{EPISODES}")

    for NUM_USERS in range(1, 11, 1):

        total_latency = 0
        total_rate = 0
        total_energy = 0
        total_sinr = 0
        total_channel = 0
        total_reward = 0  
        active_users = 0
        total_distance=0
    

        user_x = np.random.uniform(
            0,
            CELL_RADIUS,
            NUM_USERS
        )

        user_y = np.random.uniform(
            0,
            CELL_RADIUS,
            NUM_USERS
        )

        ai_x = np.random.uniform(
            0,
            CELL_RADIUS,
            NUM_AI_DEVICES
        )

        ai_y = np.random.uniform(
            0,
            CELL_RADIUS,
            NUM_AI_DEVICES
        )

        ai_speed = np.random.uniform(
            PROCESSING_SPEED_MIN,
            PROCESSING_SPEED_MAX,
            NUM_AI_DEVICES
        )

        mobility = np.random.uniform(
            MOBILITY_MIN,
            MOBILITY_MAX,
            NUM_USERS
        )
        
        user_channels = -np.ones(
          NUM_USERS,
          dtype=int
       )
         
        for u in range(NUM_USERS):

            if np.random.rand() > REQUEST_PROBABILITY:
                continue

            active_users += 1

            distances = np.sqrt(
                (user_x[u] - ai_x) ** 2 +
                (user_y[u] - ai_y) ** 2
            )

            nearest = np.argmin(distances)

            d = max(distances[nearest], 1)

            channel_gain = 1 / (
                d ** PATHLOSS_EXPONENT
            )
            interference=0
            
            state = np.array([
               d / CELL_RADIUS,
               active_users / 10,
               np.log10(interference + 1e-12)
            ])
            channel_interference = []

            for ch in CHANNELS:

                temp_interference = NOISE_POWER

                for j in range(NUM_USERS):

                    if j == u:
                        continue

                    d_interf = np.sqrt(
                        (user_x[u]-user_x[j])**2 +
                        (user_y[u]-user_y[j])**2
                    )

                    d_interf = max(d_interf,1)

                    temp_interference += (
                        TX_POWER /
                        (d_interf**PATHLOSS_EXPONENT)
                    )

                channel_interference.append(
                    float(temp_interference)
                )
            action = choose_action(state)

            ddqn_channel = CHANNELS[action]

            llm_channel = llm_allocate_channel(
                distance=d,
                active_users=active_users,
                mobility=mobility[u],
                channel_gain=channel_gain,
                channel_interference=channel_interference
            )
            # Hybrid DDQN + LLM decision
            if np.random.rand() < 0.8 :

                  selected_channel = ddqn_channel

            else:

                 selected_channel = llm_channel

            user_channels[u] = selected_channel
            
            interference = 0

            for j in range(NUM_USERS):

               if j == u:
                 continue

               if user_channels[j] != selected_channel:
                  continue

               d_interf = np.sqrt(
                 (user_x[u]-user_x[j])**2 +
                 (user_y[u]-user_y[j])**2
               )

               d_interf = max(d_interf, 1)

               interference += (
                  TX_POWER /
                  (d_interf ** PATHLOSS_EXPONENT)
               )

            next_state = np.array([
               d / CELL_RADIUS,
               active_users / 10,
               np.log10(interference + 1e-12)
            ])
            

            # ==================================
            # DQN STATE
            # ==================================

            
            selected_power_dbm = 32

            selected_power = TX_POWER
            

            signal_power = (
                selected_power *
                channel_gain
            )

            sinr = signal_power / (
                NOISE_POWER +
                interference
            )

            data_rate = (
                BANDWIDTH *
                np.log2(
                    1 + sinr
                )
            )

            throughput = data_rate

            sinr_db = (
                10 *
                np.log10(sinr)
            )

            reward = (
                 0.1 * (throughput / 1e6)
                 + sinr_db
                 - 10 * interference
                
            )
            
            
            done = False

            if u == NUM_USERS - 1:
               done = True

            transmission_delay = (
                PACKET_SIZE_BITS /
                max(data_rate, 1)
            )

            processing_delay = (
                TASK_SIZE_MB /
                ai_speed[nearest]
            )

            latency = (
                transmission_delay +
                processing_delay
            )

            energy = (
                selected_power *
                transmission_delay
            )

            ee = (
                throughput /
                max(
                    energy,
                    1e-12
                )
            )

            total_latency += latency
            total_rate += throughput
            total_energy += energy
            total_sinr += sinr
            total_distance += d
            total_channel += selected_channel
            total_reward += reward
            episode_reward += reward

            memory.append(
                (
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )
                )

            if len(memory) > BATCH_SIZE:
               replay()

            all_results.append([
                NUM_USERS,
                u + 1,
                selected_channel,
                mobility[u],
                d,
                channel_gain,
                NOISE_POWER,
                sinr,
                sinr_db,
                data_rate,
                throughput,
                transmission_delay,
                processing_delay,
                latency,
                energy,
                ee
            ])

            print("=" * 60)
            print("User :", u + 1)
            print(
                "Selected Channel:",
                selected_channel
            )
            print(
                "Reward:",
                round(
                    reward,
                    4
                )
            )
            print(
                "Mobility:",
                round(
                    mobility[u],
                    4
                )
            )
            print(
                "Distance:",
                round(
                    d,
                    4
                )
            )
            print(
                "SINR(dB):",
                round(
                    sinr_db,
                    4
                )
            )

        if active_users > 0:

            avg_latency = (
                total_latency /
                active_users
            )

            avg_rate = (
                total_rate /
                active_users
            )

            avg_energy = (
                total_energy /
                active_users
            )

            avg_ee = (
                avg_rate /
                max(
                    avg_energy,
                    1e-12
                )
            )
            avg_sinr = total_sinr / active_users
            avg_channel = total_channel / active_users
            avg_reward = total_reward / active_users   
            avg_distance = total_distance / active_users 

            if episode == (
                EPISODES - 1
            ):

                user_counts.append(
                    NUM_USERS
                )

                latency_results.append(
                    avg_latency
                )

                throughput_results.append(
                    avg_rate / 1e6
                )

                ee_results.append(
                    avg_ee / 1e6
                )
                sinr_results.append(
                   10*np.log10(avg_sinr)
                )

                channel_results.append(
                   avg_channel 
                )

                reward_results.append(
                   avg_reward
                )
                distance_results.append(
                    avg_distance
                ) 
    episode_rewards.append(
      episode_reward
    ) 
                
                
# =====================================================
# SAVE CSV
# =====================================================

columns = [
    "Total Users",
    "User ID",
    "Selected Channel",
    "Mobility",
    "Distance",
    "Channel Gain",
    "Thermal Noise",
    "SINR",
    "SINR dB",
    "Data Rate",
    "Throughput",
    "Transmission Delay",
    "Processing Delay",
    "Latency",
    "Energy",
    "Energy Efficiency"
   
]

df = pd.DataFrame(
    all_results,
    columns=columns
)

df.to_csv(
    "hybrid_ddqn_llm_6g_results.csv",
    index=False
)

print("\nCSV Saved Successfully")

# =====================================================
# GRAPH DATA
# =====================================================




# Power vs Distance

distance_power = np.arange(
    1,
    101
)

received_power = (
    TX_POWER /
    (
        distance_power **
        PATHLOSS_EXPONENT
    )
)

# =====================================================
# PLOTS
# =====================================================

fig, ax = plt.subplots(4, 2, figsize=(10,8))

# 1 LATENCY
ax[0,0].plot(
    user_counts,
    latency_results,
    marker='o'
)

ax[0,0].set_title("hybrid ddqn +llm Latency vs Users")
ax[0,0].set_xlabel("Number of Users")
ax[0,0].set_ylabel("Latency")
ax[0,0].set_xticks(user_counts)
ax[0,0].grid(True)

# 2 EE USERS
ax[0,1].plot(
    user_counts,
    ee_results,
    marker='s'
)

ax[0,1].set_title("hybrid ddqn +llm Energy Efficiency vs Users")
ax[0,1].set_xlabel("Number of Users")
ax[0,1].set_ylabel("Energy Efficiency")
ax[0,1].set_xticks(user_counts)
ax[0,1].grid(True)

# 3 SINR
ax[1,0].plot(
    user_counts,
    sinr_results,
    marker='^'
)

ax[1,0].set_title("Average SINR vs Users")
ax[1,0].set_xlabel("Number of Users")
ax[1,0].set_ylabel("SINR (dB)")
ax[1,0].set_xticks(user_counts)
ax[1,0].grid(True)

# 4 CHANNEL ALLOCATION
ax[1,1].plot(
    user_counts,
    channel_results,
    marker='d'
)

ax[1,1].set_title(
    "hybrid ddqn +llm Channel Allocation vs Users"
)

ax[1,1].set_xlabel(
    "Number of Users"
)

ax[1,1].set_ylabel(
    "Average Channel Index"
)

ax[1,1].set_xticks(
    user_counts
)

ax[1,1].grid(True)

# 5 REWARD
ax[2,0].plot(
    user_counts,
    reward_results,
    marker='o'
)

ax[2,0].set_title("hybrid ddqn +llm Reward vs Users")
ax[2,0].set_xlabel("Number of Users")
ax[2,0].set_ylabel("Average Reward")
ax[2,0].set_xticks(user_counts)
ax[2,0].grid(True)

# 6 THROUGHPUT
ax[2,1].plot(
    user_counts,
    throughput_results,
    marker='s'
)

ax[2,1].set_title("hybrid ddqn +llm Throughput vs Users")
ax[2,1].set_xlabel("Number of Users")
ax[2,1].set_ylabel("Throughput (Mbps)")
ax[2,1].set_xticks(user_counts)
ax[2,1].grid(True)

# Learning Curve
plt.figure()

plt.plot(
    episode_rewards,
    label ="DDQN"
)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("hybrid ddqn +llm Learning Curve")
plt.grid(True)

# 7 CHANNEL VS DISTANCE
ax[3,0].plot(
    distance_results,
    channel_results,
    marker='o'
)

ax[3,0].set_title(
    "Channel Allocation vs D2D Distance"
)

ax[3,0].set_xlabel(
    "Average D2D Distance (m)"
)

ax[3,0].set_ylabel(
    "Average Channel Index"
)

ax[3,0].grid(True)

ax[3,1].axis('off')

plt.tight_layout()

print("OPENING GRAPH WINDOW")

plt.show(block=True)
