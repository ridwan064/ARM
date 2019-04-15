def get_slots(max_val, n_slots):
    discrete = [i for i in range(0, max_val + n_slots, max_val // n_slots)]
    slots = [(a, b) for a, b in zip(discrete, discrete[1:])]
    return slots


def get_slot_reward(value, max_value, n_slots):
    slots = get_slots(max_value, n_slots)
    for r, slot in enumerate(slots):
        if slot[0] <= value < slot[1]:
            return r
    else:
        return n_slots

maxes = [2200, 1800, 320, 210]

for m in maxes:
    print(get_slots(m, 10))
    print(get_slot_reward(500, m, 10))
