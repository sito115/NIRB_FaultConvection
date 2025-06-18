from SALib.sample import sobol as sobol_sample
from SALib.sample import morris as morris_sample
from SALib.analyze import sobol, morris
from SALib.test_functions import Ishigami
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime



problem = {
    'num_vars': 3,
    'names': ['x1', 'x2', 'x3'], #, 'x4'],
    'bounds': [[-3.14159265359, 3.14159265359],
               [-3.14159265359, 3.14159265359],
               [-3.14159265359, 3.14159265359]],
               # [-3.14159265359, 3.14159265359]]
}

### SOBOL

param_values = sobol_sample.sample(problem, 1024)

Y = Ishigami.evaluate(param_values)
Si = sobol.analyze(problem, Y)
dfs : list[pd.DataFrame] = Si.to_df()
for idx, df in enumerate(dfs):
    df.to_csv(f'Sobol_Salib{idx}.csv')

### MORRIS
N_TRAJECTORIES = 400
NUM_LEVELS = 8
NTH_LINE = 1

param_values = morris_sample.sample(problem, N_TRAJECTORIES,
                                    num_levels=NUM_LEVELS,
                                    seed = 12300,
                                    local_optimization=True)
Y = Ishigami.evaluate(param_values)
Si, add_info = morris.analyze(problem, param_values, Y, conf_level=0.95,
                    print_to_console=True, num_levels=NUM_LEVELS, seed=12300)
result_increased, result_decreased, input_increased, input_decreased = add_info

###################################
var_index = 0

df = Si.to_df()
df['my_mu'] = np.mean((result_increased - result_decreased) / morris._compute_delta(NUM_LEVELS), axis = 1)
df['my_mu_star'] = np.mean(np.abs((result_increased - result_decreased) / morris._compute_delta(NUM_LEVELS)), axis = 1)
df['my_sigma'] = np.std((result_increased - result_decreased) / morris._compute_delta(NUM_LEVELS), axis = 1)
df.to_csv('Morris_Salib.csv')
print(df)

# current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# with open("MorrisComparison.txt", "a") as fid:
#     fid.write(f"SAlib: {current_datetime}\n")
#     fid.write(f"\t p = {NUM_LEVELS}\n")
#     fid.write(f"\t r = {N_TRAJECTORIES}\n")
#     fid.write(f"\t\t mu_star = {df.loc[:, 'mu_star'].values}\n")
#     fid.write(f"\t\t mu_     = {df.loc[:, 'mu'].values}\n")
#     fid.write(f"\t\t sigma   = {df.loc[:, 'sigma'].values}\n\n\n")


fig = go.Figure()
colors = ['blue', 'red', 'green', 'black']
for var_index in range(problem['num_vars']):
    y_start = result_increased[var_index,::NTH_LINE]
    y_end   = result_decreased[var_index,::NTH_LINE]
    x_start = input_increased[::NTH_LINE, var_index ,var_index]
    x_end   = input_decreased[::NTH_LINE, var_index ,var_index]

    x_all = np.empty(3 * len(y_start) - 1)
    y_all = np.empty(3 * len(y_start) - 1)
        
    # Fill the arrays with start and end points and np.nan in between
    y_all[::3] = y_start
    y_all[1::3] = y_end
    y_all[2::3] = np.nan  # Insert np.nan between each pair of points
    x_all[::3] = x_start
    x_all[1::3] = x_end
    x_all[2::3] = np.nan  # Insert np.nan between each pair of points
    fig.add_trace(go.Scatter(
        x=x_all,
        y=y_all,
        mode='lines',  # Show both lines and markers
        line=dict(width=0.25, color = colors[var_index]),
        name=f"X{var_index}"
    ))

    fig.add_trace(go.Scatter(
    x=np.concatenate([input_increased[:, var_index ,var_index], input_decreased[:, var_index ,var_index]]),  # Combine all points
    y=np.concatenate([result_increased[var_index,:], result_decreased[var_index,:]]),  # Combine all points
    mode='markers',
    marker=dict(size=8, color=colors[var_index], symbol='circle'),
    name=f"Points {var_index}",
    visible='legendonly',
    ))

    x = np.linspace(-np.pi, np.pi, num=100)
    ee_mean_slope = np.mean((y_end-y_start)/(x_end-x_start))
    # ee_mean_slope = df.iloc[var_index, 0]
    fig.add_trace(go.Scatter(
        x=x,
        y=x*ee_mean_slope,
        mode='lines',
        name = f'EE Mean Slope {var_index}: {ee_mean_slope:.2f}',
        line=dict(width=6, color = colors[var_index], dash='dash'),
        visible='legendonly',
        )
    )


# Customize layout
fig.update_layout(
    title=r"$\text{SAlib: Ishigami-Function} : f(x) = \sin(x_{0}) + 7 \sin(x_{1})^2 + 0.1 x^{4}_{2}\sin(x_{0})$ (with A=7, B=0.1)",
    xaxis_title="x",
    yaxis_title="f(x0,x1,x2)",
    showlegend=True,
        annotations=[
        dict(
            text=f"p={NUM_LEVELS}, r={N_TRAJECTORIES}",
            x=0.5,  # Center the subtitle
            y=1.02,  # Position below the main title
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=14, color="gray")
        )]
)

# Show plot
fig.show()