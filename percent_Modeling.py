# %%
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# ===================== 统一文件命名常量区 =====================
# 【输入文件】
INPUT_PERCENT_MODELING_DATA = '2_PERCENT_modeling_data.csv'  # 百分比建模基础数据
# 【分析结果输出】
OUTPUT_PERCENT_UNCERTAINTY_ANALYSIS = 'PERCENT_vote_uncertainty_analysis.csv'
OUTPUT_PERCENT_SOLUTION_SPACE = 'PERCENT_solution_space.csv'
# 【预测结果输出】
OUTPUT_PERCENT_PREDICTED_VOTES = '3_PERCENT_predicted_fan_votes.csv'
# 【可视化文件】
OUTPUT_PERCENT_VISUALIZATION = 'PERCENT_uncertainty_analysis.png'
# ============================================================

# %%
# 第一部分：数据导入和预处理
print("=== 第一部分：数据导入和预处理 ===")
df = pd.read_csv(INPUT_PERCENT_MODELING_DATA)
print(f"数据集形状: {df.shape}")
print(f"列名: {list(df.columns)}")
print(f"赛季分布: {sorted(df['season'].unique())}")

# 数据质量检查
print("\n数据质量检查:")
print(f"缺失值统计:")
print(df.isnull().sum())
print(f"数据类型:")
print(df.dtypes)

# 创建周次分组标识
df['season_week'] = df['season'].astype(str) + '_' + df['week'].astype(str)

# 如果没有淘汰标记，基于“最后一周”构造 eliminated_this_week
if 'eliminated_this_week' not in df.columns:
    df = df.sort_values(['celebrity_name', 'season', 'week']).copy()
    df['next_week'] = df.groupby(['celebrity_name', 'season'])['week'].shift(-1)
    df['eliminated_this_week'] = df['next_week'].isna()
    if 'placement' in df.columns:
        df.loc[df['placement'] == 1, 'eliminated_this_week'] = False
    df = df.drop(columns=['next_week'])
    print("已自动生成 eliminated_this_week 列")

print("\n第一部分完成! 数据已成功导入并预处理。")

# %%
# 第二部分：百分比票数分配函数
print("\n=== 第二部分：百分比票数分配函数 ===")

# 宏定义公共常量（统一默认参数）
N_SAMPLES = 1000       # 蒙特卡洛采样次数（测试阶段缩小）
TOTAL_VOTES = 1000000 # 总投票数
CI_LOW = 2.5          # 95%置信区间下限分位数
CI_HIGH = 97.5        # 95%置信区间上限分位数
NOISE_STD = 0.6       # 采样扰动强度（与Rank模型一致的扰动思想）

def normalize_percentages(arr):
    """将百分比裁剪并归一化到总和100"""
    arr = np.clip(np.array(arr), 0, None)
    total = arr.sum()
    if total <= 0:
        return np.ones(len(arr)) / len(arr) * 100
    return arr / total * 100

def percentage_to_votes(percentages, total_votes=TOTAL_VOTES):
    """
    将百分比转换为票数：百分比越高，票数越多
    使用组合权重确保合理分布
    """
    percentages = np.array(percentages)
    
    # 方法1: 直接使用百分比作为权重基础
    weights_direct = percentages
    
    # 方法2: 平方根平滑 - 减少极端差异
    weights_sqrt = np.sqrt(percentages)
    
    # 方法3: 对数平滑 - 进一步平滑差异
    weights_log = np.log(percentages + 1)
    
    # 组合多种权重方案，确保分配合理
    combined_weights = (weights_direct * 0.6 +
                       weights_sqrt * 0.25 +
                       weights_log * 0.15)
    
    # 归一化权重
    total_weight = np.sum(combined_weights)
    if total_weight > 0:
        normalized_weights = combined_weights / total_weight
    else:
        # 保底方案：均匀分配
        normalized_weights = np.ones(len(percentages)) / len(percentages)
    
    # 计算票数
    votes = total_votes * normalized_weights
    return votes

def estimate_fan_percentages_heuristic(weekly_data, noise_std=NOISE_STD):
    """
    启发式方法：当优化方法失败时使用
    基于评委百分比进行合理调整并加入扰动
    """
    judge_percentages = weekly_data['weekly_score_percentage'].values
    eliminated_mask = weekly_data['eliminated_this_week'].values
    
    # 初始粉丝百分比设为评委百分比（确保是numpy数组）
    fan_percentages = judge_percentages.copy()
    
    if np.sum(eliminated_mask) > 0:
        # 调整被淘汰选手的粉丝百分比，使其综合百分比最低
        eliminated_indices = np.where(eliminated_mask)[0]
        survived_indices = np.where(~eliminated_mask)[0]
        
        for elim_idx in eliminated_indices:
            # 计算幸存选手的最低综合百分比
            if len(survived_indices) > 0:
                survived_totals = np.array([judge_percentages[i] + fan_percentages[i]
                                         for i in survived_indices])
                min_survived_total = np.min(survived_totals)
                
                # 设置淘汰选手的粉丝百分比，使其综合百分比略低于最小幸存者
                required_fan_percentage = min_survived_total - judge_percentages[elim_idx] - 0.1
                fan_percentages[elim_idx] = max(0, required_fan_percentage)
    
    # 加入小扰动以形成区间
    fan_percentages = fan_percentages + np.random.normal(0, noise_std, len(fan_percentages))
    fan_percentages = normalize_percentages(fan_percentages)
    
    return fan_percentages

def sample_fan_percentages_monte_carlo(weekly_data, total_votes=TOTAL_VOTES, n_samples=N_SAMPLES, noise_std=NOISE_STD):
    """
    使用蒙特卡洛方法采样粉丝投票百分比的解空间
    参考Rank模型思路：最小化与评委差异 + 约束条件 + 随机扰动
    """
    weekly_data = weekly_data.reset_index(drop=True)
    n_players = len(weekly_data)
    judge_percentages = weekly_data['weekly_score_percentage'].values
    eliminated_mask = weekly_data['eliminated_this_week'].values
    all_samples = []

    def objective(fan_percentages):
        """目标函数：最小化粉丝百分比与评委百分比差异"""
        return np.sum((fan_percentages - judge_percentages) ** 2)
    
    def constraint_sum(fan_percentages):
        """约束：百分比之和为100"""
        return float(np.sum(fan_percentages) - 100)
    
    def constraint_elimination(fan_percentages):
        """约束条件：被淘汰选手的综合百分比必须低于所有幸存选手"""
        fan_percentages = np.array(fan_percentages)
        total_percentages = judge_percentages + fan_percentages
        
        eliminated_indices = np.where(eliminated_mask)[0]
        survived_indices = np.where(~eliminated_mask)[0]
        
        if len(eliminated_indices) == 0 or len(survived_indices) == 0:
            return 0  # 无人被淘汰或无人幸存的情况
        
        eliminated_total = np.array([total_percentages[i] for i in eliminated_indices])
        survived_total = np.array([total_percentages[i] for i in survived_indices])
        
        # 淘汰选手的最大综合百分比应小于幸存选手的最小综合百分比
        max_eliminated = np.max(eliminated_total)
        min_survived = np.min(survived_total)
        
        return float(min_survived - max_eliminated - 0.1)
    
    has_elimination_constraint = np.any(eliminated_mask) and np.any(~eliminated_mask)
    constraints = [{'type': 'eq', 'fun': constraint_sum}]
    if has_elimination_constraint:
        constraints.append({'type': 'ineq', 'fun': constraint_elimination})
    
    for _ in range(n_samples):
        try:
            # 初始猜测：评委百分比 + 随机扰动
            initial_guess = judge_percentages + np.random.normal(0, noise_std, n_players)
            initial_guess = normalize_percentages(initial_guess)
            
            result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=[(0, 100)] * n_players,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.success:
                sample_result = result.x
                # 添加轻微扰动增加多样性
                sample_result = sample_result + np.random.normal(0, noise_std * 0.5, n_players)
                sample_result = normalize_percentages(sample_result)
                all_samples.append(sample_result)
            else:
                # 失败时使用启发式方法并加扰动
                fallback = estimate_fan_percentages_heuristic(weekly_data, noise_std=noise_std)
                fallback = fallback + np.random.normal(0, noise_std * 0.5, n_players)
                fallback = normalize_percentages(fallback)
                all_samples.append(fallback)
        except Exception:
            # 保底策略：评委百分比 + 更强扰动
            fallback = judge_percentages + np.random.normal(0, noise_std * 1.5, n_players)
            fallback = normalize_percentages(fallback)
            all_samples.append(fallback)
    
    return np.array(all_samples)

def analyze_percentage_solution_space(samples, weekly_data, total_votes=TOTAL_VOTES):
    """
    分析百分比解空间的统计特性
    """
    analysis_list = []
    n_players = len(weekly_data)
    
    for i, player_name in enumerate(weekly_data['celebrity_name']):
        player_samples = samples[:, i]  # 该选手在所有样本中的粉丝百分比
        
        # 计算基本统计量
        mean_percentage = np.mean(player_samples)
        std_percentage = np.std(player_samples)
        
        # 计算置信区间 (95% 置信水平)
        ci_low = np.percentile(player_samples, CI_LOW)
        ci_high = np.percentile(player_samples, CI_HIGH)
        
        # 计算百分比分布的众数
        unique, counts = np.unique(np.round(player_samples, 1), return_counts=True)
        mode_percentage = unique[np.argmax(counts)] if len(unique) > 0 else mean_percentage
        
        # 计算票数分布
        sample_votes = []
        for sample_idx in range(samples.shape[0]):
            current_sample_percentages = samples[sample_idx, :]
            current_votes = percentage_to_votes(current_sample_percentages, total_votes)
            sample_votes.append(current_votes[i])
        
        mean_votes = np.mean(sample_votes)
        votes_ci_low = np.percentile(sample_votes, CI_LOW)
        votes_ci_high = np.percentile(sample_votes, CI_HIGH)
        
        # 计算置信区间宽度（千票为单位）
        ci_width_thousands = (votes_ci_high - votes_ci_low) / 1000
        
        analysis_list.append({
            'celebrity_name': player_name
            , 'judge_percentage': float(weekly_data.iloc[i]['weekly_score_percentage'])
            , 'eliminated_this_week': bool(weekly_data.iloc[i]['eliminated_this_week'])
            , 'mean_fan_percentage': float(mean_percentage)
            , 'std_fan_percentage': float(std_percentage)
            , 'fan_percentage_ci_low': float(ci_low)
            , 'fan_percentage_ci_high': float(ci_high)
            , 'fan_percentage_ci_width': float(ci_high - ci_low)
            , 'mode_fan_percentage': float(mode_percentage)
            , 'mean_fan_votes': float(mean_votes)
            , 'fan_votes_ci_low': float(votes_ci_low)
            , 'fan_votes_ci_high': float(votes_ci_high)
            , 'fan_votes_ci_width': float(ci_width_thousands)
            , 'combined_percentage_mean': float(weekly_data.iloc[i]['weekly_score_percentage'] + mean_percentage)
            , 'certainty_index': float(1 / (std_percentage + 0.1))
        })
    
    return pd.DataFrame(analysis_list)

print("第二部分完成! 百分比票数分配函数已创建。")

# %%
# 第二点五部分：扰动强度校准（测试阶段）
print("\n=== 第二点五部分：扰动强度校准 ===")

def calibrate_noise_std(df, noise_grid, n_samples=80, max_weeks=25, random_state=42):
    """
    根据淘汰一致性与区间宽度，校准扰动强度
    """
    weeks = df[['season', 'week']].drop_duplicates()
    if len(weeks) > max_weeks:
        weeks = weeks.sample(max_weeks, random_state=random_state)
    
    results = []
    
    for noise_std in noise_grid:
        consistency_list = []
        ci_width_list = []
        
        for _, w in weeks.iterrows():
            weekly_data = df[(df['season'] == w['season']) & (df['week'] == w['week'])].copy()
            if len(weekly_data) < 2:
                continue
            
            samples = sample_fan_percentages_monte_carlo(
                weekly_data, n_samples=n_samples, noise_std=noise_std
            )
            analysis = analyze_percentage_solution_space(samples, weekly_data)
            
            eliminated = analysis[analysis['eliminated_this_week'] == True]
            survived = analysis[analysis['eliminated_this_week'] == False]
            if not eliminated.empty and not survived.empty:
                ok = eliminated['combined_percentage_mean'].max() < survived['combined_percentage_mean'].min()
                consistency_list.append(1 if ok else 0)
            
            ci_width_list.append(analysis['fan_percentage_ci_width'].mean())
        
        if ci_width_list:
            results.append({
                'noise_std': noise_std,
                'consistency_rate': np.mean(consistency_list) if consistency_list else np.nan,
                'avg_fan_pct_ci_width': np.mean(ci_width_list),
                'weeks_evaluated': len(ci_width_list)
            })
    
    return pd.DataFrame(results)

# 运行校准（测试阶段）
NOISE_GRID = [0.2, 0.4, 0.6, 0.8, 1.0]
calib_summary = calibrate_noise_std(df, NOISE_GRID, n_samples=80, max_weeks=25)
print("\n扰动强度校准结果:")
print(calib_summary.round(4).to_string(index=False))

# %%
# 第三部分：基于解空间的批量处理
print("\n=== 第三部分：基于解空间的批量处理 ===")

def process_all_weeks_percentage(df, total_votes=TOTAL_VOTES, n_samples=N_SAMPLES):
    """
    处理所有周次，进行百分比解空间分析
    """
    all_uncertainty_results = []
    space_characteristics = []
    
    unique_weeks = df[['season', 'week']].drop_duplicates()
    
    print(f"总共需要处理 {len(unique_weeks)} 个周次")
    
    for idx, week_info in enumerate(unique_weeks.iterrows()):
        _, week_info = week_info
        season, week = week_info['season'], week_info['week']
        weekly_data = df[(df['season'] == season) & (df['week'] == week)].copy()
        
        print(f"正在处理赛季 {season} 第 {week} 周 ({idx+1}/{len(unique_weeks)})")
        
        if len(weekly_data) < 2:
            print(f"  跳过：选手数量不足 ({len(weekly_data)})")
            continue
        
        try:
            # 1. 采样解空间
            print(f"  开始采样解空间 ({n_samples} 个样本)")
            samples = sample_fan_percentages_monte_carlo(weekly_data, total_votes, n_samples)
            print(f"  采样完成，样本形状: {samples.shape}")
            
            # 2. 分析解空间特性
            print(f"  开始分析解空间")
            week_analysis = analyze_percentage_solution_space(samples, weekly_data, total_votes)
            week_analysis['season'] = season
            week_analysis['week'] = week
            
            # 3. 计算本周解空间的整体确定性
            avg_std = week_analysis['std_fan_percentage'].mean()
            avg_ci_width = (week_analysis['fan_percentage_ci_high'] -
                          week_analysis['fan_percentage_ci_low']).mean()
            
            space_characteristics.append({
                'season': season,
                'week': week,
                'n_players': len(weekly_data),
                'avg_std_fan_percentage': avg_std,
                'avg_ci_width': avg_ci_width,
                'solution_space_compactness': 1 / (avg_std + 0.1),
            })
            
            all_uncertainty_results.append(week_analysis)
            print(f"  处理完成")
            
        except Exception as e:
            print(f"处理赛季{season}第{week}周时出错: {e}")
            continue
    
    if all_uncertainty_results:
        final_uncertainty = pd.concat(all_uncertainty_results, ignore_index=True)
        space_df = pd.DataFrame(space_characteristics)
        return final_uncertainty, space_df
    else:
        return pd.DataFrame(), pd.DataFrame()

# 执行处理
print("开始百分比解空间分析...")
uncertainty_results, space_stats = process_all_weeks_percentage(df, n_samples=N_SAMPLES)

if not uncertainty_results.empty:
    print(f"解空间分析完成! 共分析 {len(space_stats)} 个周次")
    
    # 计算整体确定性指标
    overall_avg_std = space_stats['avg_std_fan_percentage'].mean()
    overall_compactness = space_stats['solution_space_compactness'].mean()
    
    print(f"整体百分比标准差: {overall_avg_std:.3f}")
    print(f"解空间紧凑性指数: {overall_compactness:.3f}")
    
    # 保存结果
    uncertainty_results.to_csv(OUTPUT_PERCENT_UNCERTAINTY_ANALYSIS, index=False)
    space_stats.to_csv(OUTPUT_PERCENT_SOLUTION_SPACE, index=False)
    print("结果已保存为CSV文件")
    print(f"- {OUTPUT_PERCENT_UNCERTAINTY_ANALYSIS}")
    print(f"- {OUTPUT_PERCENT_SOLUTION_SPACE}")
else:
    print("解空间分析失败，无结果生成")

print("\n第三部分完成!")

# %%
# 第四部分：可视化分析
print("\n=== 第四部分：可视化分析 ===")

def _with_suffix(path, suffix):
    """在文件名末尾追加后缀（保留扩展名）"""
    if '.' in path:
        base, ext = path.rsplit('.', 1)
        return f"{base}{suffix}.{ext}"
    return f"{path}{suffix}"

def plot_percentage_uncertainty_analysis(uncertainty_results, space_stats):
    """
    绘制百分比不确定性分析结果（拆分为4张图）
    """
    if uncertainty_results.empty or space_stats.empty:
        print("无可视化数据，跳过绘图")
        return
    
    # 1. 粉丝投票估计的置信区间示例（用区间线显示范围）
    try:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sample_week_data = uncertainty_results[
            (uncertainty_results['season'] == 3) & 
            (uncertainty_results['week'] == 1.0)
        ].head(8)
        
        if not sample_week_data.empty:
            y_pos = range(len(sample_week_data))
            for i, (_, row) in enumerate(sample_week_data.iterrows()):
                ax.hlines(y=i, xmin=row['fan_votes_ci_low'], xmax=row['fan_votes_ci_high'],
                          color='steelblue', alpha=0.8, linewidth=3)
                ax.plot(row['mean_fan_votes'], i, 'o', color='black', markersize=4)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(sample_week_data['celebrity_name'], fontsize=8)
            ax.set_xlabel('粉丝投票数估计（区间）')
            ax.set_title('粉丝投票估计的置信区间（赛季3第1周）')
            ax.grid(True, axis='x')
        else:
            ax.set_title('粉丝投票估计的置信区间（赛季3第1周）')
            ax.text(0.5, 0.5, '无可用数据', ha='center', va='center', transform=ax.transAxes)
        
        fig.tight_layout()
        path1 = _with_suffix(OUTPUT_PERCENT_VISUALIZATION, '_1')
        fig.savefig(path1, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"图1已保存: {path1}")
    except Exception as e:
        print(f"绘制图1时出错: {e}")
    
    # 2. 评委百分比 vs 粉丝百分比散点图
    try:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sample_data = uncertainty_results.head(100)
        ax.scatter(sample_data['judge_percentage'], sample_data['mean_fan_percentage'],
                   alpha=0.6, c=sample_data['eliminated_this_week'].map({True: 'red', False: 'blue'}))
        ax.set_xlabel('评委百分比')
        ax.set_ylabel('粉丝百分比估计')
        ax.set_title('评委百分比 vs 粉丝百分比估计')
        ax.grid(True, alpha=0.3)
        lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', alpha=0.3)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        fig.tight_layout()
        path2 = _with_suffix(OUTPUT_PERCENT_VISUALIZATION, '_2')
        fig.savefig(path2, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"图2已保存: {path2}")
    except Exception as e:
        print(f"绘制图2时出错: {e}")
    
    # 3. 综合百分比分布
    try:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        combined_data = uncertainty_results['combined_percentage_mean']
        ax.hist(combined_data, bins=30, alpha=0.7, color='green')
        ax.axvline(combined_data.mean(), color='red', linestyle='--',
                   label=f'平均值: {combined_data.mean():.1f}%')
        ax.set_xlabel('综合百分比（评委+粉丝）')
        ax.set_ylabel('频数')
        ax.set_title('综合百分比分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        path3 = _with_suffix(OUTPUT_PERCENT_VISUALIZATION, '_3')
        fig.savefig(path3, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"图3已保存: {path3}")
    except Exception as e:
        print(f"绘制图3时出错: {e}")
    
    # 4. 各赛季平均确定性比较
    try:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        season_certainty = uncertainty_results.groupby('season')['certainty_index'].mean()
        seasons = season_certainty.index.astype(int)
        values = season_certainty.values
        ax.bar(seasons, values, color='lightcoral', alpha=0.7)
        ax.set_xlabel('赛季')
        ax.set_ylabel('平均确定性指数')
        ax.set_title('各赛季估计确定性比较')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        path4 = _with_suffix(OUTPUT_PERCENT_VISUALIZATION, '_4')
        fig.savefig(path4, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"图4已保存: {path4}")
    except Exception as e:
        print(f"绘制图4时出错: {e}")

# 执行可视化
if not uncertainty_results.empty and not space_stats.empty:
    print("开始绘制可视化图表...")
    plot_percentage_uncertainty_analysis(uncertainty_results, space_stats)
else:
    print("数据为空，无法进行可视化")

print("\n第四部分完成!")

# %%
# 第四点五部分：文本方式查看预测区间
print("\n=== 第四点五部分：文本预测区间展示 ===")

def show_text_prediction_intervals(uncertainty_results, season=3, week=1.0, top_n=10):
    """
    用文本方式展示指定周次的预测区间
    """
    week_data = uncertainty_results[
        (uncertainty_results['season'] == season) & 
        (uncertainty_results['week'] == week)
    ].copy()
    
    if week_data.empty:
        print(f"未找到赛季{season}第{week}周的数据")
        return
    
    # 选取区间宽度最大的前N位，方便观察范围
    week_data['ci_width'] = week_data['fan_votes_ci_high'] - week_data['fan_votes_ci_low']
    week_data = week_data.sort_values('ci_width', ascending=False).head(top_n)
    
    display_cols = [
        'celebrity_name', 'judge_percentage', 'mean_fan_percentage',
        'fan_percentage_ci_low', 'fan_percentage_ci_high',
        'mean_fan_votes', 'fan_votes_ci_low', 'fan_votes_ci_high',
        'eliminated_this_week'
    ]
    print(f"\n赛季{season}第{week}周 - 预测区间（Top {top_n}）:")
    print(week_data[display_cols].round(2).to_string(index=False))

def show_single_player_interval(uncertainty_results, season=3, week=1.0):
    """
    随机挑一名选手，展示其预测区间
    """
    week_data = uncertainty_results[
        (uncertainty_results['season'] == season) & 
        (uncertainty_results['week'] == week)
    ].copy()
    if week_data.empty:
        print(f"未找到赛季{season}第{week}周的数据")
        return
    
    sample_row = week_data.sample(1, random_state=42).iloc[0]
    print("\n随机选手预测区间:")
    print(f"选手: {sample_row['celebrity_name']}")
    print(f"评委百分比: {sample_row['judge_percentage']:.2f}%")
    print(f"粉丝百分比均值: {sample_row['mean_fan_percentage']:.2f}%")
    print(f"粉丝百分比区间: [{sample_row['fan_percentage_ci_low']:.2f}%, {sample_row['fan_percentage_ci_high']:.2f}%]")
    print(f"粉丝票数均值: {sample_row['mean_fan_votes']:.0f}")
    print(f"粉丝票数区间: [{sample_row['fan_votes_ci_low']:.0f}, {sample_row['fan_votes_ci_high']:.0f}]")
    print(f"是否被淘汰: {sample_row['eliminated_this_week']}")

# 执行文本展示
if not uncertainty_results.empty:
    show_text_prediction_intervals(uncertainty_results, season=3, week=1.0, top_n=10)
    show_single_player_interval(uncertainty_results, season=3, week=1.0)
else:
    print("无可用的不确定性结果")

# %%
# 第五部分：生成预测票数文件
print("\n=== 第五部分：生成预测票数文件 ===")

def generate_predicted_votes_for_merge(uncertainty_results):
    """
    从不确定性分析结果中提取预测票数，生成供合并使用的文件
    """
    if uncertainty_results.empty:
        print("无不确定性分析结果，无法生成预测票数文件")
        return pd.DataFrame()
    
    # 选择需要的列
    predicted_votes = uncertainty_results[[
        'season', 'week', 'celebrity_name', 'mean_fan_votes', 'mean_fan_percentage'
    ]].copy()
    
    # 重命名列
    predicted_votes = predicted_votes.rename(columns={
        'mean_fan_votes': 'predicted_fan_votes',
        'mean_fan_percentage': 'predicted_fan_percentage'
    })
    
    # 确保数据类型一致
    predicted_votes['season'] = pd.to_numeric(predicted_votes['season'], errors='coerce')
    predicted_votes['week'] = pd.to_numeric(predicted_votes['week'], errors='coerce')
    
    # 保存文件
    predicted_votes.to_csv(OUTPUT_PERCENT_PREDICTED_VOTES, index=False)
    
    print(f"预测票数文件生成完成!")
    print(f"记录数量: {len(predicted_votes)}")
    print(f"赛季范围: {predicted_votes['season'].min()} - {predicted_votes['season'].max()}")
    print(f"周次范围: {predicted_votes['week'].min()} - {predicted_votes['week'].max()}")
    print(f"选手数量: {predicted_votes['celebrity_name'].nunique()}")
    print(f"文件已保存为: {OUTPUT_PERCENT_PREDICTED_VOTES}")
    
    return predicted_votes

# 生成预测票数文件
if not uncertainty_results.empty:
    print("正在生成预测票数文件...")
    predicted_votes_df = generate_predicted_votes_for_merge(uncertainty_results)
    
    # 显示统计信息
    print(f"\n预测票数统计:")
    print(f"平均预测票数: {predicted_votes_df['predicted_fan_votes'].mean():.0f}")
    print(f"预测票数范围: {predicted_votes_df['predicted_fan_votes'].min():.0f} - {predicted_votes_df['predicted_fan_votes'].max():.0f}")
    print(f"平均粉丝百分比: {predicted_votes_df['predicted_fan_percentage'].mean():.2f}%")
else:
    print("未找到可用的不确定性分析结果")

print("\n第五部分完成!")

# %%
# 第六部分：模型验证和总结
print("\n=== 第六部分：模型验证和总结 ===")

def validate_percentage_model(uncertainty_results):
    """
    验证百分比模型的合理性
    """
    if uncertainty_results.empty:
        print("无数据可验证")
        return
    
    print("模型验证结果:")
    
    # 1. 检查淘汰选手的综合百分比是否低于幸存选手
    eliminated_data = uncertainty_results[uncertainty_results['eliminated_this_week'] == True]
    survived_data = uncertainty_results[uncertainty_results['eliminated_this_week'] == False]
    
    if not eliminated_data.empty and not survived_data.empty:
        avg_eliminated_combined = eliminated_data['combined_percentage_mean'].mean()
        avg_survived_combined = survived_data['combined_percentage_mean'].mean()
        
        print(f"淘汰选手平均综合百分比: {avg_eliminated_combined:.2f}%")
        print(f"幸存选手平均综合百分比: {avg_survived_combined:.2f}%")
        
        if avg_eliminated_combined < avg_survived_combined:
            print("✓ 淘汰规则验证通过: 淘汰选手综合百分比低于幸存选手")
        else:
            print("✗ 淘汰规则验证失败: 淘汰选手综合百分比异常")
    
    # 2. 检查百分比分布合理性
    fan_percentages = uncertainty_results['mean_fan_percentage']
    print(f"粉丝百分比分布: {fan_percentages.min():.1f}% - {fan_percentages.max():.1f}%")
    print(f"粉丝百分比标准差: {fan_percentages.std():.2f}%")
    
    # 3. 检查票数分布合理性
    fan_votes = uncertainty_results['mean_fan_votes']
    total_predicted_votes = fan_votes.sum()
    expected_total = TOTAL_VOTES * len(uncertainty_results[['season', 'week']].drop_duplicates())
    
    print(f"预测总票数: {total_predicted_votes:,.0f}")
    print(f"预期总票数: {expected_total:,.0f}")
    print(f"票数偏差: {(total_predicted_votes - expected_total) / expected_total * 100:.2f}%")
    
    return True

# 执行模型验证
if not uncertainty_results.empty:
    validation_result = validate_percentage_model(uncertainty_results)
    
    print(f"\n=== 百分比机制模型总结 ===")
    print("模型特点:")
    print("1. 基于百分比机制设计，符合赛季3-27的实际规则")
    print("2. 使用组合权重确保票数合理分配")
    print("3. 蒙特卡洛采样考虑淘汰约束和百分比求和约束")
    print("4. 输出包含置信区间和不确定性指标")
    
    print(f"\n分析范围:")
    print(f"- 赛季数量: {uncertainty_results['season'].nunique()}")
    print(f"- 周次数量: {uncertainty_results[['season', 'week']].drop_duplicates().shape[0]}")
    print(f"- 选手记录: {len(uncertainty_results)}")
    
    print(f"\n生成文件:")
    print("1. percentage_fan_vote_uncertainty_analysis.csv - 详细不确定性分析")
    print("2. percentage_solution_space_characteristics.csv - 解空间特性")
    print("3. percentage_predicted_fan_votes.csv - 预测票数文件")
    print("4. percentage_uncertainty_analysis.png - 可视化图表")
else:
    print("无法进行模型验证")

print("\n所有代码执行完成! 🎉")

# %%



