"""
    看一下soul的loss横坐标是epoch还是iter？
"""
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score


def plotting_loss(traintime, training_losses=None, validation_losses=None):
        """
            考虑是否可以将loss和f2_score放到一张图里
            plotting train | validtation loss
        """
        if training_losses:
            epoch_count = range(1, len(training_losses)+1)
            plt.plot(epoch_count, training_losses, 'r--')
            plt.title(['Training Loss'])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(f'output/loss/{traintime}_training_loss.jpg')
        if validation_losses:
            epoch_count = range(1, len(validation_losses)+1)
            plt.plot(epoch_count, validation_losses, 'b--')
            plt.title(['Validation Loss'])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(f'output/loss/{traintime}_validation_loss.jpg')


def plotting_f2_score(traintime, training_f2_score=None, validation_f2_score=None):
        """
            plotting train | validation f2_score
        """
        if training_f2_score:
            epoch_count = range(1, len(training_f2_score)+1)
            plt.plot(epoch_count, training_f2_score, 'r--')
            plt.title(['Training F2 Score'])
            plt.xlabel('Epoch')
            plt.ylabel('F2 Score')
            plt.savefig(f'output/f2_score/{traintime}_training_f2_score.jpg')
        if validation_f2_score:
            epoch_count = range(1, len(validation_f2_score)+1)
            plt.plot(epoch_count, validation_f2_score, 'b--')
            plt.title(['Validation F2 Score'])
            plt.xlabel('Epoch')
            plt.ylabel('F2 Score')
            plt.savefig(f'output/f2_score/{traintime}_validation_f2_score.jpg')


def calculate_F2_score(y_pred, y_true):
    """
        可以定义成utils
    """
    y_pred = [1 if i>=0.5 else 0 for i in y_pred]
    precision = precision_score(y_pred, y_true)
    recall = recall_score(y_pred, y_true)
    F2_score = 5 * precision * recall / (4 * precision + recall)

    return F2_score