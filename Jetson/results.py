import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv

def getValues(path):
    Episode, Score = [], []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            Episode.append(int(row[0]))
            Score.append(float(row[1]))
    return Episode, Score

Episode, Score = getValues("score_history.csv")
Episode = np.array(Episode)
Score = np.array(Score)
plt.plot(Episode, Score)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Training Progress')
plt.savefig('traning_progress.png', dpi=300, bbox_inches='tight')