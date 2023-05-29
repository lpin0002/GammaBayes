from utils import inverse_transform_sampling, bkgdist, makedist, edisp, eaxis_mod, log10eaxis
from scipy import integrate, special, interpolate, stats
import os, time, random, sys, numpy as np, matplotlib.pyplot as plt, chime, warnings, corner.corner as corner
from tqdm import tqdm
from BFCalc.BFInterp import DM_spectrum_setup
from matplotlib.backends.backend_agg import FigureCanvasAgg


try:
    stemidentifier = sys.argv[1]
except:
    stemidentifier = time.strftime("%d%m%H")
    
try:
    lastidx = int(sys.argv[2])
except:
    lastidx = 29

try:
    firstidx = int(sys.argv[3])
except:
    firstidx = 1



def plotonerun(identifier):
    currentdirecyory = os.getcwd()
    stemdirectory = currentdirecyory+f'/data/{identifier}'
    
    rundirs = [x[0] for x in os.walk(stemdirectory)][1:]

    params               = np.load(f"{rundirs[0]}/params.npy")
    
    totalevents          = int(params[1,1])
    truelambda           = float(params[1,0])
    truelogmass          = float(params[1,2])

    for rundir in rundirs[1:]:
            tempparams           = np.load(f"{rundir}/params.npy")
            totalevents          += int(tempparams[1,1])
    
    
    recyclingresults     = np.load(f'{stemdirectory}/recyclingresults.npy', allow_pickle=True)

    recyclingresults = recyclingresults.item()
    runsamples = recyclingresults.samples_equal()


    figure = corner(
                runsamples,
                levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                labels=[r"log$_{10}$ $m_\chi$", r"$\lambda$"],
                show_titles=True,
                title_kwargs={"fontsize": 12},
                bins = [25,25],
                truths=[truelogmass, truelambda],
                labelpad=-0.1,
                tick_kwargs={'rotation':90},
                color='#0072C1',
                truth_color='tab:orange',
                plot_density=0, 
                plot_datapoints=True, 
                fill_contours=True,
                max_n_ticks=7,
                hist_kwargs=dict(density=True),
                smooth=0.9,
                # smooth1d=0.9
    )
    figure.set_size_inches(6,6)
    figure.set_dpi(100)
    #plt.tight_layout()
    plt.close()
    return figure

figurelist = []

for i in range(firstidx, lastidx+1):
    try:
        figurelist.append(plotonerun(stemidentifier+f"{i}"))
    except:
        pass

numberrange = np.arange(1,9)
squarenumbers = numberrange**2
# Define the number of rows and columns in the grid
closestsquare = numberrange[np.abs(squarenumbers-len(figurelist)).argmin()]
print("Closest square", closestsquare)
num_cols = int(np.round(len(figurelist)/closestsquare))+1
print("num_cols", num_cols)
num_rows = closestsquare

# Create the main figure and axes for the grid
fig, axes = plt.subplots(num_rows, num_cols, dpi=500, figsize=(12,8))

# Flatten the axes array to simplify indexing
axes = axes.flatten()

# Iterate over the figure objects and add them to the grid
for i, fig_obj in enumerate(figurelist):
    # Activate the current axis
    ax = axes[i]
    ax.axis('off')  # Hide the axis labels and ticks

    try:
        fig_obj.tight_layout()  # Adjust the layout of the individual plot
        # Create a canvas and render the figure object as an image
        
        canvas = FigureCanvasAgg(fig_obj)
        canvas.draw()
        
        # Extract the image from the canvas and display it in the current axis
        image = canvas.buffer_rgba()
        ax.imshow(image)
    except:
        pass

# Remove any unused axes in the grid
for j in range(len(figurelist), num_rows * num_cols):
    fig.delaxes(axes[j])

# Adjust the spacing between the subplots
fig.subplots_adjust(hspace=1/(num_rows*3), wspace=1/(num_cols*3))
fig.tight_layout()
# Display the grid of plots
plt.savefig(f"Figures/{stemidentifier}gridofresults.pdf")
plt.show()