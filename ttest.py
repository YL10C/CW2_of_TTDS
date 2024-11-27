from scipy.stats import ttest_rel

# 假设我们要比较系统1和系统2在P@10上的差异
system1_p10 = [metrics for metrics in system_metrics_1['P@10']]
system2_p10 = [metrics for metrics in system_metrics_2['P@10']]

# 进行配对t检验
t_stat, p_value = ttest_rel(system1_p10, system2_p10)

# 判断p值是否小于0.05
if p_value < 0.05:
    print("系统1在P@10上显著优于系统2")
else:
    print("系统1在P@10上不显著优于系统2")
