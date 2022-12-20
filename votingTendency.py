####################################################################################
#                                  Import packages
####################################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches                   # for the legend
from statsmodels.graphics.mosaicplot import mosaic      # to create a mosaic plot

####################################################################################
#                                  Functions
####################################################################################

def mult_stacked_bar(table, colors, counts = [], figsize = [10,8], filename = 'plot.png', ax = None,
                     legend_title = '', xlabel = '',):
    
    if ax != None:
        ax = table.plot.bar(stacked = True, rot = 0, fontsize = 16, width = 0.9,
                    figsize = figsize, color = colors, ax = ax)
    else:
        ax = table.plot.bar(stacked = True, rot = 0, fontsize = 16, width = 0.9,
                    figsize = figsize, color = colors)

    # remove all of the spines 
    [ax.spines[i].set_visible(False) for i in ax.spines]

    # remove x tick marks
    ax.tick_params(axis = 'x', length = 0)

    # remove y-axis
    ax.tick_params(axis = 'y', length = 0)
    ax.get_yaxis().set_visible(True)

    n_rows = table.shape[0]
    n_cols = table.shape[1]

    # put labels inside of the bars
    for i in range(n_rows): # for each row: 0, 1
        # new row means new bar
        prev = 0
        for j in range(n_cols): # for each column: 0, 1
            current = table.iloc[i, j]
            ypos = current / 2 + prev
            if current != 0: # if there is some data
                if round(current*100) >= 10: # if the data is equal to or more than 10%
                    ax.text(i, ypos, f'{current*100:.0f}%', # put the label in this style
                           horizontalalignment = 'center', verticalalignment = 'center',
                           color = 'white', fontsize = 9, weight = 'bold')
            prev += current # update which bar we've filled

    # put labels above the bars with number of individuals who answered the survey
    if len(counts) > 0:
        for i in range(n_rows): # row numbers: 0, 1
            ax.text(i, 1.05, f'n={counts.iloc[i]:.0f}', fontsize = 12, horizontalalignment = 'center')
            #ax.text(i, 1.05, 'n=' + str(counts[i]), fontsize = 15, horizontalalignment = 'center')

    # label the x axis
    plt.xlabel(xlabel, fontsize = 16, labelpad = 10)
    
    # locate the legend
    ax.legend(bbox_to_anchor = [1.025, 0.5], loc = 'center left',
               title = legend_title,
               title_fontsize = 18, frameon = False, markerfirst = False, fontsize = 14)
    
    # save figure
    plt.savefig(filename, bbox_inches = 'tight')
    #plt.show()
    
    return ax, plt

def cross_line_plot(df, income_cat, ax = None, color = None):
    # subset the dataframe for the wanted income category
    df = df[df['income_cat'] == income_cat]    
    
    # get the percentage of voted or not voted for past electinos for each race accordingly
    yes_w1, no_w1 = df[df['race'] == 'White']['Q27_6'].value_counts()
    yes_w2, no_w2 = df[df['race'] == 'White']['Q27_5'].value_counts()
    yes_w3, no_w3 = df[df['race'] == 'White']['Q27_4'].value_counts()
    yes_w4, no_w4 = df[df['race'] == 'White']['Q27_3'].value_counts()
    yes_w5, no_w5 = df[df['race'] == 'White']['Q27_2'].value_counts()
    yes_w6, no_w6 = df[df['race'] == 'White']['Q27_1'].value_counts()
    w_total = len(df[df['race'] == 'White']['Q27_1'])
    
    yes_b1, no_b1 = df[df['race'] == 'Black']['Q27_6'].value_counts()
    yes_b2, no_b2 = df[df['race'] == 'Black']['Q27_5'].value_counts()
    yes_b3, no_b3 = df[df['race'] == 'Black']['Q27_4'].value_counts()
    yes_b4, no_b4 = df[df['race'] == 'Black']['Q27_3'].value_counts()
    yes_b5, no_b5 = df[df['race'] == 'Black']['Q27_2'].value_counts()
    yes_b6, no_b6 = df[df['race'] == 'Black']['Q27_1'].value_counts()
    b_total = len(df[df['race'] == 'Black']['Q27_1'])
    
    yes_h1, no_h1 = df[df['race'] == 'Hispanic']['Q27_6'].value_counts()
    yes_h2, no_h2 = df[df['race'] == 'Hispanic']['Q27_5'].value_counts()
    yes_h3, no_h3 = df[df['race'] == 'Hispanic']['Q27_4'].value_counts()
    yes_h4, no_h4 = df[df['race'] == 'Hispanic']['Q27_3'].value_counts()
    yes_h5, no_h5 = df[df['race'] == 'Hispanic']['Q27_2'].value_counts()
    yes_h6, no_h6 = df[df['race'] == 'Hispanic']['Q27_1'].value_counts()
    h_total = len(df[df['race'] == 'Hispanic']['Q27_1'])
    
    yes_o1, no_o1 = df[df['race'] == 'Other/Mixed']['Q27_6'].value_counts()
    yes_o2, no_o2 = df[df['race'] == 'Other/Mixed']['Q27_5'].value_counts()
    yes_o3, no_o3 = df[df['race'] == 'Other/Mixed']['Q27_4'].value_counts()
    yes_o4, no_o4 = df[df['race'] == 'Other/Mixed']['Q27_3'].value_counts()
    yes_o5, no_o5 = df[df['race'] == 'Other/Mixed']['Q27_2'].value_counts()
    yes_o6, no_o6 = df[df['race'] == 'Other/Mixed']['Q27_1'].value_counts()
    o_total = len(df[df['race'] == 'Other/Mixed']['Q27_1'])
    
    # get current axis
    if ax == None:
        ax = plt.gca()
    
    # create constant variables for the plot
    line_width = 2
    ms = 15
    markerew = 4
    linestyle = '--'
    
    # create line plots for people who VOTED in the past elections as percentage
    # the line plots for people who did NOT VOTED in the past elections were commented out
    ax.plot([0, 1, 2, 3, 4, 5], [yes_w1/w_total, yes_w2/w_total, yes_w3/w_total, yes_w4/w_total, yes_w5/w_total, yes_w6/w_total], 
            label = "White: Voted", color = color[0], marker = 'o', lw = line_width, markersize = ms, mfc = 'white', mew = markerew, ls = linestyle)
    #ax.plot([0, 1, 2, 3, 4, 5], [no_w1/w_total, no_w2/w_total, no_w3/w_total, no_w4/w_total, no_w5/w_total, no_w6/w_total], 
    #        label = "White: Did not voted", color = color[0], marker = 's', lw = line_width, markersize = ms, mfc = 'white', mew = markerew, ls = linestyle)
    ax.plot([0, 1, 2, 3, 4, 5], [yes_b1/b_total, yes_b2/b_total, yes_b3/b_total, yes_b4/b_total, yes_b5/b_total, yes_b6/b_total], 
            label = "Black: Voted", color = color[1], marker = 'o', lw = line_width, markersize = ms, mfc = 'white', mew = markerew, ls = linestyle)
    #ax.plot([0, 1, 2, 3, 4, 5], [no_b1/b_total, no_b2/b_total, no_b3/b_total, no_b4/b_total, no_b5/b_total, no_b6/b_total], 
    #        label = "Black: Did not voted", color = color[1], marker = 's', lw = line_width, markersize = ms, mfc = 'white', mew = markerew, ls = linestyle)
    ax.plot([0, 1, 2, 3, 4, 5], [yes_h1/h_total, yes_h2/h_total, yes_h3/h_total, yes_h4/h_total, yes_h5/h_total, yes_h6/h_total], 
            label = "Hispanic: Voted", color = color[2], marker = 'o', lw = line_width, markersize = ms, mfc = 'white', mew = markerew, ls = linestyle)
    #ax.plot([0, 1, 2, 3, 4, 5], [no_h1/h_total, no_h2/h_total, no_h3/h_total, no_h4/h_total, no_h5/h_total, no_h6/h_total], 
    #        label = "Hispanic: Did not voted", color = color[2], marker = 's', lw = line_width, markersize = ms, mfc = 'white', mew = markerew, ls = linestyle)
    ax.plot([0, 1, 2, 3, 4, 5], [yes_o1/o_total, yes_o2/o_total, yes_o3/o_total, yes_o4/o_total, yes_o5/o_total, yes_o6/o_total], 
            label = "Other/Mixed: Voted", color = color[3], marker = 'o', lw = line_width, markersize = ms, mfc = 'white', mew = markerew, ls = linestyle)
    #ax.plot([0, 1, 2, 3, 4, 5], [no_o1/o_total, no_o2/o_total, no_o3/o_total, no_o4/o_total, no_o5/o_total, no_o6/o_total], 
    #        label = "Other/Mixed: Did not voted", color = color[3], marker = 's', lw = line_width, markersize = ms, mfc = 'white', mew = markerew, ls = linestyle)
    
    # set the limit to be constant
    plt.ylim([0, 1])

    return ax, plt

####################################################################################
#                                   Data Processing
####################################################################################

# Import Data
# About the data: https://github.com/fivethirtyeight/data/blob/master/non-voters/nonvoters_codebook.pdf
data = pd.read_csv('nonvoters_data.csv')

# Subset the data for the mosaic plot
mosaic_data = data.copy()
mosaic_data = mosaic_data[['race','income_cat']]
mosaic_data.sort_values(['income_cat'])

# Subset data for each race category
other_mixed = data.loc[data['race'] == 'Other/Mixed']
hispanic = data.loc[data['race'] == 'Hispanic']
black = data.loc[data['race'] == 'Black']
white = data.loc[data['race'] == 'White']

# Subset data for each income category
rich = data.loc[data['income_cat'] == '$125k or more']
middle = data.loc[data['income_cat'] == '$75-125k']
low = data.loc[data['income_cat'] == '$40-75k']
poor = data.loc[data['income_cat'] == 'Less than $40k']

# Create and process a new dataframe for a time-series plot
time = data[['Q27_1', 'Q27_2', 'Q27_3', 'Q27_4', 'Q27_5', 'Q27_6', 'ppage', 'race', 'income_cat', 'voter_category']]
# Drop rows that has -1 in any of its answers
time.drop(time[(time['Q27_1'] == -1) | (time['Q27_2'] == -1) | (time['Q27_3'] == -1) | (time['Q27_4'] == -1) | (time['Q27_5'] == -1) | (time['Q27_6'] == -1)].index, inplace=True)



####################################################################################
#                                  Data Visualization
####################################################################################

# VISUALIZATION No. 1
# Mosaic plot describing the composition of the sample

# create cross table
table = pd.crosstab(data['race'], data['income_cat'])
table = table.reindex(['White', 'Black', 'Hispanic', 'Other/Mixed'], columns = ['Less than $40k', '$40-75k', '$75-125k', '$125k or more'])

# set the figure size and its layout
plt.rcParams["figure.figsize"] = [12.00, 8.00]
plt.rcParams["figure.autolayout"] = True

# create lambda function for the label parameter
label = lambda k: ""

# set the color of the mosaic plot
cols = {('White', '$125k or more'):'#3ED30E', ('White', '$75-125k'):'#83DE66',
        ('White','$40-75k' ):'#A4E191', ('White', 'Less than $40k'):'#C2E7B6',
        ('Black', '$125k or more'):'#9218FF', ('Black', '$75-125k'):'#A745FF',
        ('Black','$40-75k' ):'#BE74FF', ('Black', 'Less than $40k'):'#D6A9FF',
        ('Hispanic', '$125k or more'):'#FF8300', ('Hispanic', '$75-125k'):'#FFA23F',
        ('Hispanic','$40-75k' ):'#FFBB74', ('Hispanic', 'Less than $40k'):'#FFD4A5',
        ('Other/Mixed', '$125k or more'):'#FFDC00', ('Other/Mixed', '$75-125k'):'#FFE22D',
        ('Other/Mixed','$40-75k' ):'#FFE95B', ('Other/Mixed', 'Less than $40k'):'#FFEF8A'}

# create the mosaic plot
fig, rects = mosaic(table.stack(), labelizer = label, axes_label = False, 
       properties = lambda key: {'color': cols[key]}, gap=0.01)

# get the axis and remove its spines
ax = plt.gca()
[ax.spines[i].set_visible(False) for i in ax.spines]

# give title to the plot
plt.text(0.5, 1.025, 'The Number of Individuals in Sample by Each Race and Income Group',
         fontsize = 16, rotation = 0, horizontalalignment = 'center', verticalalignment = 'center') 

# for each cell in the mosaic plot
for i in rects:
    # calculate its center
    x = (rects[i][0] + rects[i][2]/2)
    y = (rects[i][1] + rects[i][3]/2)

    # label the number of people for the corresponding cell
    plt.text(x, y, table.loc[i],
         fontsize = 14, rotation = 0, horizontalalignment = 'center', verticalalignment = 'center')
    
    # label the cell's race and income only once at corresponding locations using if statements
    if i == ('White', 'Less than $40k'):
        plt.text(x, -0.03, i[0],
             fontsize = 16, rotation = 0, horizontalalignment = 'center', verticalalignment = 'center')
        plt.text(-0.06, y, 'Less\nthan $40k',
             fontsize = 16, rotation = 0, horizontalalignment = 'center', verticalalignment = 'center')
    elif i == ('White', '$40-75k'):
        plt.text(-0.06, y, i[1],
             fontsize = 16, rotation = 0, horizontalalignment = 'center', verticalalignment = 'center')
    elif i == ('White', '$75-125k'):
        plt.text(-0.06, y, i[1],
             fontsize = 16, rotation = 0, horizontalalignment = 'center', verticalalignment = 'center')
    elif i == ('White', '$125k or more'):
        plt.text(-0.06, y, '$125k\nor more',
             fontsize = 16, rotation = 0, horizontalalignment = 'center', verticalalignment = 'center')
    elif i == ('Black', 'Less than $40k'):
        plt.text(x, -0.03, i[0],
             fontsize = 16, rotation = 0, horizontalalignment = 'center', verticalalignment = 'center')
    elif i == ('Hispanic', 'Less than $40k'):
        plt.text(x, -0.03, i[0],
             fontsize = 16, rotation = 0, horizontalalignment = 'center', verticalalignment = 'center')
    elif i == ('Other/Mixed', 'Less than $40k'):
        plt.text(x, -0.05, 'Other/\nMixed',
             fontsize = 16, rotation = 0, horizontalalignment = 'center', verticalalignment = 'center')

# label the x and y axis
plt.xlabel('Race', fontsize = 16, labelpad = 50)
plt.text(-0.17, 0.5, 'Income',
             fontsize = 16, rotation = 0, horizontalalignment = 'center', verticalalignment = 'center')

# save the figure
plt.savefig('mosaic.png', bbox_inches = 'tight')



# VISUALIZATION No. 2
# Stacked bar plot showing the voting tendency of sample in each race and income groups

# Create a figure with four subplots
f, axes = plt.subplots(4,1, figsize = [10, 40], gridspec_kw={'hspace':0.2, 'wspace':0.1})

# Set the color for the visualization
colors = ['cornflowerblue', 'mediumpurple', 'salmon']

# Create a cross table for income category '$125k or more' for their voting tendency,
# reorder the columns and rows, count the number of response for each categories,
# and create the bar plot for a subplot.
table = pd.crosstab(rich['race'], rich['voter_category'], normalize = 'index')
table = table.reindex(['White', 'Black', 'Hispanic', 'Other/Mixed'], columns = ['always', 'sporadic', 'rarely/never'])
counts = rich['race'].value_counts().reindex(['White', 'Black', 'Hispanic', 'Other/Mixed'])
mult_stacked_bar(table, colors, counts = counts, ax = axes[0], figsize = [10, 10])

# Repeat the process above with income category '$75-125k' 
table = pd.crosstab(middle['race'], middle['voter_category'], normalize = 'index')
table = table.reindex(['White', 'Black', 'Hispanic', 'Other/Mixed'], columns = ['always', 'sporadic', 'rarely/never'])
counts = middle['race'].value_counts().reindex(['White', 'Black', 'Hispanic', 'Other/Mixed'])
mult_stacked_bar(table, colors, counts = counts, ax = axes[1], figsize = [10, 10])

# Repeat the process above with income category '$40-75K' 
table = pd.crosstab(low['race'], low['voter_category'], normalize = 'index')
table = table.reindex(['White', 'Black', 'Hispanic', 'Other/Mixed'], columns = ['always', 'sporadic', 'rarely/never'])
counts = low['race'].value_counts().reindex(['White', 'Black', 'Hispanic', 'Other/Mixed'])
mult_stacked_bar(table, colors, counts = counts, ax = axes[2], figsize = [10, 10])

# Repeat the process above with income category '$Less than 40K' 
table = pd.crosstab(poor['race'], poor['voter_category'], normalize = 'index')
table = table.reindex(['White', 'Black', 'Hispanic', 'Other/Mixed'], columns = ['always', 'sporadic', 'rarely/never'])
counts = poor['race'].value_counts().reindex(['White', 'Black', 'Hispanic', 'Other/Mixed'])
mult_stacked_bar(table, colors, counts = counts, ax = axes[3], xlabel = 'Race', figsize = [10, 10])

# remove legend for the first three subplots
axes[0].get_legend().remove()
axes[1].get_legend().remove()
axes[2].get_legend().remove()

# remove x-axis for the first three subplots
axes[0].get_xaxis().set_visible(False)
axes[1].get_xaxis().set_visible(False)
axes[2].get_xaxis().set_visible(False)

# remove ticks and tick labels on the y-axis for all subplots
axes[0].get_yaxis().set_ticks([])
axes[1].get_yaxis().set_ticks([])
axes[2].get_yaxis().set_ticks([])
axes[3].get_yaxis().set_ticks([])

# label each subplot's income category
axes[0].set_ylabel('$125k or\nmore', fontsize = 14, rotation = 0)
axes[1].set_ylabel('$75-125k', fontsize = 14, rotation = 0)
axes[2].set_ylabel('$40-75k', fontsize = 14, rotation = 0)
axes[3].set_ylabel('Less than\n$40k', fontsize = 14, rotation = 0)

# use the legend from the last subplot and place it at an adequate location
axes[3].legend(bbox_to_anchor = [0.95, 2.125], loc = 'center left',
               labels = ['Always', 'Sporadic', 'Rarely/Never'],
               title = 'Voting Tendency',
               title_fontsize = 16, frameon = False, markerfirst = False, fontsize = 14)

# create title for the first subplot and place it at an adequate location
axes[0].set_title("For Each Income Group,\nHow Does the Voting Tendency Differ Between Races?", fontsize = 18, pad = 20)

# create label for the y-axis using the second subplot
axes[1].text(-1.25, 0, "Income", fontsize = 16, #weight = 'bold', 
                    rotation = 0, horizontalalignment = 'center', verticalalignment = 'center')

# save the figure
plt.savefig('stackedBarPlot.png', bbox_inches = 'tight')



# VISUALIZATION No. 3
# Line plot showing the percentage of people voted in the recent six elections by race and income

# Create a figure with four subplots
f, axes = plt.subplots(4,1, figsize = [15,15], gridspec_kw={'hspace':0.15, 'wspace':0.1}, sharex = True)

# Set the color for the visualization
color = ['limegreen', 'rebeccapurple','darkorange', 'gold']

# Create line plots for each income category
cross_line_plot(time, '$125k or more', ax = axes[0], color = color)
cross_line_plot(time, '$75-125k', ax = axes[1], color = color)
cross_line_plot(time, '$40-75k', ax = axes[2], color = color)
cross_line_plot(time, 'Less than $40k', ax = axes[3], color = color)

# Create constant variables for the y tick
yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
yticklabels = ['0%', '20%', '40%', '60%', '80%', '100%']

# Remove tick marks
axes[0].tick_params(axis = 'both', length = 0)
axes[1].tick_params(axis = 'both', length = 0)
axes[2].tick_params(axis = 'both', length = 0)
axes[3].tick_params(axis = 'both', length = 0)

# Set places for the y ticks
axes[0].set_yticks(ticks = yticks)
axes[1].set_yticks(ticks = yticks)
axes[2].set_yticks(ticks = yticks)
axes[3].set_yticks(ticks = yticks)

# Label the y ticks
axes[0].set_yticklabels(labels = yticklabels, fontsize = 15, rotation = 0)
axes[1].set_yticklabels(labels = yticklabels, fontsize = 15, rotation = 0)
axes[2].set_yticklabels(labels = yticklabels, fontsize = 15, rotation = 0)
axes[3].set_yticklabels(labels = yticklabels, fontsize = 15, rotation = 0)

# Label the y axis
axes[0].set_ylabel('$125k or\nmore', fontsize = 20, rotation = 0, labelpad = 70)
axes[1].set_ylabel('$75-125k', fontsize = 20, rotation = 0, labelpad = 70)
axes[2].set_ylabel('$40-75k', fontsize = 20, rotation = 0, labelpad = 70)
axes[3].set_ylabel('Less than\n$40k', fontsize = 20, rotation = 0, labelpad = 70)

# Label x ticks
axes[3].set_xticklabels(labels = ['', '2008\nPresidential', '2010\nCongressional',
                                  '2012\nPresidential', '2014\nCongressional',
                                  '2016\nPresidential', '2018\nCongressional'], 
                        fontsize = 20)#, rotation = 45)

# label x axis
axes[3].set_xlabel(xlabel = "Election", fontsize = 22, labelpad = 20)

# Label the y axis
axes[1].text(-1, .35, "Income", fontsize = 22, #weight = 'bold', 
                    rotation = 0, horizontalalignment = 'center', verticalalignment = 'center') 

# Draw grid lines for all plot
axes[0].grid(axis='both')
axes[1].grid(axis='both')
axes[2].grid(axis='both')
axes[3].grid(axis='both')

# Reset the limit of the y range
axes[0].set_ylim(0.4, 1)
axes[1].set_ylim(0.4, 1)
axes[2].set_ylim(0.4, 1)
axes[3].set_ylim(0.4, 1)

# Move the tick labels from the left to right
axes[0].yaxis.tick_right()
axes[1].yaxis.tick_right()
axes[2].yaxis.tick_right()
axes[3].yaxis.tick_right()

# Title the plot
axes[0].set_title("For Each Income Group,\nHow Many Voters by Race Groups Voted in the Recent Six Elections?", fontsize = 25, pad = 20)

# Define a lambda function to create marker for the legend
f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none", markersize = 20)[0]

# Create handle for the legend
handles = [f("s", color[i]) for i in range(len(color))]

# Create legend
axes[3].legend(handles = handles, labels = ['White', 'Black', 'Hispanic', 'Other/Mixed'],
               bbox_to_anchor = [1.3, 2.7], fontsize = 18, frameon = False,
               title = "People who Voted", title_fontsize = 20)

# Save figure
plt.savefig('time.png', bbox_inches = 'tight')



# VISUALIZATION No. 4
# Stacked bar plot showing whether people are planning to vote for the next presidential election by race

# Create a figure with four subplots
f, axes = plt.subplots(4,1, figsize = [10, 40], gridspec_kw={'hspace':0.15, 'wspace':0.1})

# Set the color for the visualization
colors = ['cornflowerblue', 'mediumpurple', 'salmon']

# From the rich table created above, drop rows that has -1 for Q21 column
# Create a cross table for their future voting tendency and their current voting tendency,
# reorder the columns and rows and create the bar plot for a subplot.
rich_dropped = rich.drop(rich[(rich['Q21'] == -1)].index)
table = pd.crosstab(rich_dropped['race'], rich_dropped['Q21'], normalize = 'index')
table = table.reindex(['White', 'Black', 'Hispanic', 'Other/Mixed'], columns = [1, 3, 2])
counts = rich_dropped['race'].value_counts().reindex(['White', 'Black', 'Hispanic', 'Other/Mixed'])
mult_stacked_bar(table, colors, counts = counts, ax = axes[0], figsize = [10, 10])

# Repeat the process above with the middle table
middle_dropped = middle.drop(middle[(middle['Q21'] == -1)].index)
table = pd.crosstab(middle_dropped['race'], middle_dropped['Q21'], normalize = 'index')
table = table.reindex(['White', 'Black', 'Hispanic', 'Other/Mixed'], columns = [1, 3, 2])
counts = middle_dropped['race'].value_counts().reindex(['White', 'Black', 'Hispanic', 'Other/Mixed'])
mult_stacked_bar(table, colors, counts = counts, ax = axes[1], figsize = [10, 10])

# Repeat the process above with the low table
low_dropped = low.drop(low[(low['Q21'] == -1)].index)
table = pd.crosstab(low_dropped['race'], low_dropped['Q21'], normalize = 'index')
table = table.reindex(['White', 'Black', 'Hispanic', 'Other/Mixed'], columns = [1, 3, 2])
counts = low_dropped['race'].value_counts().reindex(['White', 'Black', 'Hispanic', 'Other/Mixed'])
mult_stacked_bar(table, colors, counts = counts, ax = axes[2], figsize = [10, 10])

# Repeat the process above with the poor table
poor_dropped = poor.drop(poor[(poor['Q21'] == -1)].index)
table = pd.crosstab(poor_dropped['race'], poor_dropped['Q21'], normalize = 'index')
table = table.reindex(['White', 'Black', 'Hispanic', 'Other/Mixed'], columns = [1, 3, 2])
counts = poor_dropped['race'].value_counts().reindex(['White', 'Black', 'Hispanic', 'Other/Mixed'])
mult_stacked_bar(table, colors, counts = counts, ax = axes[3], figsize = [10, 10], xlabel = 'Race')

# remove legend for the first three subplots
axes[0].get_legend().remove()
axes[1].get_legend().remove()
axes[2].get_legend().remove()

# remove x-axis for the first three subplots
axes[0].get_xaxis().set_visible(False)
axes[1].get_xaxis().set_visible(False)
axes[2].get_xaxis().set_visible(False)

# remove ticks and tick labels on the y-axis for all subplots
axes[0].get_yaxis().set_ticks([])
axes[1].get_yaxis().set_ticks([])
axes[2].get_yaxis().set_ticks([])
axes[3].get_yaxis().set_ticks([])

# label each subplot's income category
axes[0].set_ylabel('$125k or\nmore', fontsize = 14, rotation = 0)
axes[1].set_ylabel('$75-125k', fontsize = 14, rotation = 0)
axes[2].set_ylabel('$40-75k', fontsize = 14, rotation = 0)
axes[3].set_ylabel('Less than\n$40k', fontsize = 14, rotation = 0)

# replace the legend
axes[3].legend(bbox_to_anchor = [0.95, 2.125], loc = 'center left',
               labels = ['Always', 'Sporadic', 'Rarely/Never'],
               title = 'Voting Tendency',
               title_fontsize = 16, frameon = False, markerfirst = False, fontsize = 14)

# remove tick marks
axes[3].tick_params(axis = 'both', length = 0)

# label the x ticks while removing the x label
axes[3].set_xticklabels(['White', 'Black', 'Hispanic', 'Other/Mixed'], fontsize = 16, rotation = 0)

# set the title for the plot
axes[0].set_title("For Each Income Group, How Many Voters by Race Groups\nAre Planning to Vote in the 2020 Presidential Election?", fontsize = 18, pad = 20)

# name the x axis
axes[1].text(-1.25, 0, "Income", fontsize = 16, #weight = 'bold', 
                    rotation = 0, horizontalalignment = 'center', verticalalignment = 'center') 

# save figure
plt.savefig('futureVoting.png', bbox_inches = 'tight')