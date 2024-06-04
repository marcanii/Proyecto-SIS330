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

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Learning Curve')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.savefig(figure_file)

Episode, Score = getValues("score_history.csv")
Episode = np.array(Episode)
Score = np.array(Score)
plot_learning_curve(Episode, Score, "learning_curve.png")

# plt.plot(Episode, Score)
# plt.xlabel('Episode')
# plt.ylabel('Score')
# plt.title('Training Progress')
# plt.savefig('traning_progress.png', dpi=300, bbox_inches='tight')