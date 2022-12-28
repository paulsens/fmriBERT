import random
forward_count = 0
reverse_count = 0
seq_len = 5
direction = random.randint(0, 1)
if direction == 0:
    # has not been reversed
    forward_count += 1
    start_idx = 0
    end_idx = seq_len
    incr = 1
else:
    # has been reversed, direction==1
    reverse_count += 1
    start_idx = seq_len - 1
    end_idx = -1
    incr = -1

for j in range(start_idx, end_idx, incr):
    print(j)