import os
import pandas as pd
import re
import os
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==================== 配置区 ====================
DATA_DIR = Path("./data")                     # 数据所在文件夹（当前目录）
SAVE_DIR = Path("./convergence_curves")   # 图片保存目录
SAVE_DIR.mkdir(exist_ok=True)

LOG_Y = True          # 强烈建议开启，对CEC函数收敛跨度极大
DPI = 300
FIG_SIZE = (9, 6)
# ================================================

def plot_separate_convergence_curves():
    # 1. 查找所有 dim=30 的 csv 文件
    csv_files = sorted(DATA_DIR.glob("*dim=30*.csv"))
    if not csv_files:
        print("错误：当前目录未找到包含 'dim=30' 的 CSV 文件！")
        return

    print(f"共发现 {len(csv_files)} 个 dim=30 的收敛记录，开始逐个绘图...\n")

    for csv_path in csv_files:
        print(f"正在处理：{csv_path.name}")

        # 2. 读取 Best_cost 列（自动跳过 inf）
        df = pd.read_csv(csv_path)
        # 通常第二列是 Best_cost
        if "Best_cost" in df.columns:
            data = df["Best_cost"]
        else:
            data = df.iloc[:, 1]  # 备用方案：第二列

        # 转为数值，inf 自动变成 NaN，再前向填充（极少数情况）
        best_cost = pd.to_numeric(data, errors='coerce')
        best_cost = best_cost.ffill().bfill()  # 确保无 NaN

        iterations = range(len(best_cost))

        # 3. 提取函数名（支持 F12014、F1、F23 等多种命名）
        filename = csv_path.stem
        if "'s_" in filename:
            func_name = filename.split("'s_")[0]        # 如 F12014
        elif "dim" in filename:
            func_name = filename.split("_dim")[0]        # 备用
        else:
            func_name = filename

        # 4. 开始绘图
        plt.figure(figsize=FIG_SIZE)
        plt.plot(iterations, best_cost, color='blue', linewidth=2.0)

        plt.title(f"Convergence Curve of {func_name} (30D)", fontsize=16, pad=15)
        plt.xlabel("Iteration", fontsize=14)
        plt.ylabel("Best Cost", fontsize=14)

        if LOG_Y:
            plt.yscale('log')
            plt.ylabel("Best Cost (log scale)", fontsize=14)

        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

        # 5. 保存图片（文件名干净）
        # save_name = f"{func_name}_convergence_30D.png"
        # save_path = SAVE_DIR / save_name
        # plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        # plt.close()  # 关闭图，释放内存

        # print(f"已保存：{save_path}")

    print(f"\n所有收敛曲线绘制完成！共 {len(csv_files)} 张图，已保存至：{SAVE_DIR.resolve()}")

# ============== 运行 ==============
if __name__ == "__main__":
    plot_separate_convergence_curves()
# def extract_cec2014_last_values(data_dir="data"):
#     results = []
#
#     for file_name in os.listdir(data_dir):
#         # 只处理包含 "2014" 的文件
#         if "2014" not in file_name:
#             continue
#
#         if file_name.endswith(".csv"):
#             # 提取 F 函数编号 (例如 F1, F2, F16)
#             match = re.match(r"F(\d+)", file_name)
#             if not match:
#                 print(f"无法解析函数编号: {file_name}")
#                 continue
#
#             func_id = int(match.group(1))
#
#             # 加载 CSV
#             file_path = os.path.join(data_dir, file_name)
#             df = pd.read_csv(file_path)
#
#             # 最后一次迭代 (最后一行最后一列)
#             last_value = df.iloc[-1].iloc[-1]
#
#             results.append((func_id, file_name, last_value))
#
#     # 按函数编号排序
#     results.sort(key=lambda x: x[0])
#
#     # 打印结果
#     print("CEC2014 函数结果（按 F1→F.. 排序）：\n")
#     for func_id, file_name, value in results:
#         print(f"F{func_id:02d}: {file_name} → 最后迭代值 = {value}")
#
#     # 保存总结
#     summary = pd.DataFrame(
#         [{"Function_ID": func_id, "File": file_name, "Last_Value": value}
#          for func_id, file_name, value in results]
#     )
#     summary.to_csv("CEC2014_Last_Values_Sorted.csv", index=False)
#
#     print("\n已保存：CEC2014_Last_Values_Sorted.csv")
#
#     return results
