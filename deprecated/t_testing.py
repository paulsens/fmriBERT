import scipy.stats as stats


k=12
val_accs = [12.25, 12.75, 12.25, 13.25, 12.5, 13, 13.875, 13.25, 11.375, 11.625, 12, 14.375]

assert len(val_accs)==k, f"for {k}-fold cross validation, only got {len(val_accs)} accuracies.\n"

t_statistic, p_value = stats.ttest_1samp(a=val_accs, popmean=11)
print(t_statistic , p_value)