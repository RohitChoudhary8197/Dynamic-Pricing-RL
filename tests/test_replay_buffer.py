from memory.replay_buffer import ReplayBuffer

buffer = ReplayBuffer(capacity=5)

for i in range(5):

    buffer.add(
        state=[i, i],
        action=i,
        reward=i * 100,
        next_state=[i + 1, i + 1],
        done=False
    )

print("Memory Size :", len(buffer))

states, actions, rewards, next_states, dones = buffer.sample(3)

print("\nStates")
print(states)

print("\nActions")
print(actions)

print("\nRewards")
print(rewards)