# library: tqdm
# version: 4.28
# extra_dependencies: []
from tqdm import tqdm


def infinite():
    i = 0
    while True:
        yield i
        i += 1
        if i == 1000:
            return


# Define the total in sol_dict['total'] and use it.
sol_dict = {"total": 0}
sol_dict["total"] = None
progress_bar = tqdm(infinite(), total=sol_dict["total"])
for progress in progress_bar:
    progress_bar.set_description(f"Processing {progress}")
