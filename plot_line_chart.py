import matplotlib.pyplot as plt
import numpy as np

def plot_modal_ablation(configs, ccc_values, rmse_values, title_top=None, title_bottom=None, save_path=None):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # 左轴
    color_ccc = 'red'
    ax1.set_xlabel(title_bottom, fontsize=18)                 # X轴标签字体大小
    ax1.set_ylabel('CCC', color=color_ccc, fontsize=18)       # 左Y轴标签字体大小
    line1 = ax1.plot(configs, ccc_values, color=color_ccc, marker='o', 
                     linestyle='-', linewidth=2, markersize=8, label='CCC')
    ax1.tick_params(axis='y', labelcolor=color_ccc, labelsize=16)  # 左Y轴刻度字体大小
    ax1.tick_params(axis='x', labelsize=16)                      # X轴刻度字体大小（旋转前）
    ax1.set_ylim(bottom=min(0, min(ccc_values)*0.9), top=max(ccc_values)*1.1)
    
    # 右轴
    ax2 = ax1.twinx()
    color_rmse = 'blue'
    ax2.set_ylabel('RMSE', color=color_rmse, fontsize=18)     # 右Y轴标签字体大小
    line2 = ax2.plot(configs, rmse_values, color=color_rmse, marker='s', 
                     linestyle='--', linewidth=2, markersize=8, label='RMSE')
    ax2.tick_params(axis='y', labelcolor=color_rmse, labelsize=16)  # 右Y轴刻度字体大小
    
    # 图例
    # lines = line1 + line2
    # labels = [l.get_label() for l in lines]
    # ax1.legend(lines, labels, loc='upper right', fontsize=16)  # 图例字体大小
    
    # 旋转X轴刻度
    plt.xticks(rotation=45)
    
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    if title_top:
        plt.title(title_top, fontsize=14)                      # 标题字体大小
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"图片已保存至: {save_path}")
    else:
        plt.show()

# ==================== 示例用法 ====================
if __name__ == "__main__":
    configs = ['(5, 5)', '(10, 10)', '(15, 15)', '(20, 20)', '(25, 25)', '(30, 30)']
    # configs = ['1', '2', '3', '4', '5', '6']

    ccc_example = [0.513, 0.634, 0.570, 0.599, 0.611, 0.536]
    rmse_example = [5.54, 4.85, 5.63, 4.96, 5.32, 5.24]

    # ccc_example = [0.490, 0.567, 0.593, 0.634, 0.646, 0.605]
    # rmse_example = [5.61, 5.39, 5.51, 4.85, 4.95, 5.50]
    
    # 绘制
    plot_modal_ablation(
        configs=configs,
        ccc_values=ccc_example,
        rmse_values=rmse_example,
        title_top=None,
        title_bottom='Size of Windows (Np, Nf)',
        # title_bottom='Number of heads in Graph Transformer',
        save_path='./picture/Size_of_Windows.png'   # 可改为None直接显示
        # save_path='./picture/Number_of_heads_in_GraphTransformer.png'  
    )