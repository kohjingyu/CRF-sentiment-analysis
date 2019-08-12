import glob

result_dir = "results/preds/"
dataset = "ES" # ES or EN
files = [f for f in glob.glob(result_dir + f"*{dataset}*.out")]

best_f1 = 0
best_file = ""

for f in files:
    total_f1 = 0
    count = 0
    with open(f, "r") as rf:
        lines = rf.readlines()
        f_needles = ["Entity  F:", "Sentiment  F:"]

        for line in lines:
            for needle in f_needles:
                if needle in line:
                    value = line[len(needle):]
                    total_f1 += float(value)
                    count += 1

    assert(count == 2) # Only 2 F1 scores
    avg_f1 = total_f1 / count
    if avg_f1 > best_f1:
        best_f1 = avg_f1
        best_file = f


print(f"Best F1 score is {best_f1} for file {best_file}")
