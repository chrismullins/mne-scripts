for time in times:
    fig = evoked.plot_topomap(times=time, ch_type='eeg', vmin=-8.0, vmax=8.0, show=False)
    fig.savefig(filename="gif3/image-{:04d}.png".format(i))
    i = i+1
    plt.close(fig)
