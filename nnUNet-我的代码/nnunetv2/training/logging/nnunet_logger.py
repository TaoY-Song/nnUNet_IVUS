import matplotlib
from batchgenerators.utilities.file_and_folder_operations import join

matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt


class nnUNetLogger(object):
    """
    This class is really trivial. Don't expect cool functionality here. This is my makeshift solution to problems
    arising from out-of-sync epoch numbers and numbers of logged loss values. It also simplifies the trainer class a
    little

    YOU MUST LOG EXACTLY ONE VALUE PER EPOCH FOR EACH OF THE LOGGING ITEMS! DONT FUCK IT UP
    """
    def __init__(self, verbose: bool = False):
        self.my_fantastic_logging = {
            'mean_fg_dice': list(),
            'ema_fg_dice': list(),
            'dice_per_class_or_region': list(),
            'train_losses': list(),
            'val_losses': list(),
            'lrs': list(),
            'epoch_start_timestamps': list(),
            'epoch_end_timestamps': list()
        }
        self.verbose = verbose
        # shut up, this logging is great

    def log(self, key, value, epoch: int):
        """
        sometimes shit gets messed up. We try to catch that here
        """
        assert key in self.my_fantastic_logging.keys() and isinstance(self.my_fantastic_logging[key], list), \
            'This function is only intended to log stuff to lists and to have one entry per epoch'

        if self.verbose: print(f'logging {key}: {value} for epoch {epoch}')

        if len(self.my_fantastic_logging[key]) < (epoch + 1):
            self.my_fantastic_logging[key].append(value)
        else:
            assert len(self.my_fantastic_logging[key]) == (epoch + 1), 'something went horribly wrong. My logging ' \
                                                                       'lists length is off by more than 1'
            print(f'maybe some logging issue!? logging {key} and {value}')
            self.my_fantastic_logging[key][epoch] = value

        # handle the ema_fg_dice special case! It is automatically logged when we add a new mean_fg_dice
        if key == 'mean_fg_dice':
            new_ema_pseudo_dice = self.my_fantastic_logging['ema_fg_dice'][epoch - 1] * 0.9 + 0.1 * value \
                if len(self.my_fantastic_logging['ema_fg_dice']) > 0 else value
            self.log('ema_fg_dice', new_ema_pseudo_dice, epoch)

    def plot_progress_png(self, output_folder):
        # 调整子图布局为4行1列，增加图像高度
        sns.set(font_scale=2.5)
        fig, ax_all = plt.subplots(4, 1, figsize=(30, 72))  # 高度从54调整为72
        
        # 第一子图：损失和Dice分数（保持原有）
        ax = ax_all[0]
        ax2 = ax.twinx()
        x_values = list(range(len(self.my_fantastic_logging['train_losses'])))
        ax.plot(x_values, self.my_fantastic_logging['train_losses'], color='b', ls='-', label="loss_tr", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_losses'], color='r', ls='-', label="loss_val", linewidth=4)
        ax2.plot(x_values, self.my_fantastic_logging['mean_fg_dice'], color='g', ls='dotted', label="pseudo dice", linewidth=3)
        ax2.plot(x_values, self.my_fantastic_logging['ema_fg_dice'], color='g', ls='-', label="pseudo dice (mov. avg.)", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("pseudo dice")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.2, 1))

        # 第二子图：epoch持续时间（保持原有）
        ax = ax_all[1]
        ax.plot(x_values, [i - j for i, j in zip(self.my_fantastic_logging['epoch_end_timestamps'],
                                                 self.my_fantastic_logging['epoch_start_timestamps'])], 
                color='b', ls='-', label="epoch duration", linewidth=4)
        ylim = [0] + [ax.get_ylim()[1]]
        ax.set(ylim=ylim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        ax.legend(loc=(0, 1))

        # 第三子图：学习率（保持原有）
        ax = ax_all[2]
        ax.plot(x_values, self.my_fantastic_logging['lrs'], color='b', ls='-', label="learning rate", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))

        # 新增第四子图：不确定性曲线
        ax = ax_all[3]
        if 'train_uncertainty' in self.my_fantastic_logging and len(self.my_fantastic_logging['train_uncertainty']) > 0:
            ax.plot(x_values, self.my_fantastic_logging['train_uncertainty'], color='m', ls='-', label="train_uncert", linewidth=4)
        if 'val_uncertainty' in self.my_fantastic_logging and len(self.my_fantastic_logging['val_uncertainty']) > 0:
            ax.plot(x_values, self.my_fantastic_logging['val_uncertainty'], color='c', ls='--', label="val_uncert", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("uncertainty")
        ax.legend(loc=(0, 1))
        ax.set_ylim(0, 1)  # 假设不确定性范围在0-1之间

        plt.tight_layout(pad=5.0)  # 增加间距防止标签重叠
        fig.savefig(join(output_folder, "progress.png"))
        plt.close()

    def get_checkpoint(self):
        return self.my_fantastic_logging

    def load_checkpoint(self, checkpoint: dict):
        self.my_fantastic_logging = checkpoint
