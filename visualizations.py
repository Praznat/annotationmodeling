
from matplotlib import pyplot as plt


def visualize_embeddings(stan_data, opt, sim_df=None, preds={}):
    from sklearn.decomposition import PCA
    def userset(data):
        result = set(data["u1s"]).union(set(data["u2s"]))
        return result
    sddf = pd.DataFrame(stan_data)
    item_userset = sddf.groupby("items").apply(userset)
    for i, iue in enumerate(opt["item_user_errors"]):
        users = item_userset.get(i+1)
        if users is None or len(users) < 3:
            continue
        dist_from_truth = opt["dist_from_truth"][i]
        print("item", str(i+1))
        print(sddf[sddf["items"]==i+1][["u1s", "u2s", "distances"]])
        if len(iue[0]) > 2:
            embeddings = PCA(n_components=2).fit_transform(iue)
        embeddings = np.array([embeddings[u-1] for u in users])
        dists = [dist_from_truth[u-1] for u in users]
        skills = [opt["uerr"][u-1] for u in users]
        scale = np.max(np.abs(embeddings)) * 1.05
        plt.scatter(embeddings[:,0], embeddings[:,1])
        for ui, emb in enumerate(embeddings):
            plt.plot([0,emb[0]], [0,emb[1]], "b")
            plt.annotate(str(list(users)[ui]) + ":" + str(np.round(dists[ui],2)) + ":" + str(np.round(skills[ui],2)), emb)
        if sim_df is not None:
            preds.get(i)
            sim_df[sim_df.topic_item==0]
        plt.xlim(-scale, scale)
        plt.ylim(-scale, scale)
        plt.show()


def plot_vectorrange(vr, color="b", alpha=0.3, text=None, ax=None):
    if ax is None:
        ax = plt
    ax.plot([vr.start_vector[0], vr.end_vector[0]], [vr.start_vector[1], vr.start_vector[1]], color, alpha=alpha)
    ax.plot([vr.start_vector[0], vr.start_vector[0]], [vr.start_vector[1], vr.end_vector[1]], color, alpha=alpha)
    ax.plot([vr.start_vector[0], vr.end_vector[0]], [vr.end_vector[1], vr.end_vector[1]], color, alpha=alpha)
    ax.plot([vr.end_vector[0], vr.end_vector[0]], [vr.start_vector[1], vr.end_vector[1]], color, alpha=alpha)
    if text is not None:
        x, y = vr.centroid()
        ax.text(x, y, text, color=color)

def plot_seqrange(vr, color="b", alpha=0.3, text=None, ax=None):
    if ax is None:
        ax = plt
    yT = 0
    yB = 10
    ax.plot([vr.start_vector[0], vr.end_vector[0]], [yT, yT], color, alpha=alpha)
    ax.plot([vr.start_vector[0], vr.start_vector[0]], [yT, yB], color, alpha=alpha)
    ax.plot([vr.start_vector[0], vr.end_vector[0]], [yB, yB], color, alpha=alpha)
    ax.plot([vr.end_vector[0], vr.end_vector[0]], [yT, yB], color, alpha=alpha)
    if text is not None:
        ax.text(vr.start_vector[0], yT, text, color=color)

def plot_vrimg(vr, alpha=0.3, ax=None):
    import matplotlib.image as mpimg
    if ax is None:
        ax = plt
    img = mpimg.imread('grey.png')
    try:
        imgplot = ax.imshow(img, aspect="auto", extent=(vr.start_vector[0], vr.end_vector[0], vr.start_vector[1], vr.end_vector[1]))
        imgplot.set_alpha(alpha)
    except:
        pass

def plot_annos(data, golddict={}):
    vrs = [vr for annotation in data["annotation"] for vr in annotation]
    for vr in vrs:
        plot_vectorrange(vr)
    item = data["item"].values[0]
    gold = golddict.get(item)
    if gold is not None:
        for gvr in golddict.get(item):
            # plot_vectorrange(gvr, "yo--", alpha=1)
            plot_vrimg(gvr, alpha=0.5)
    plt.title(data["item"].values[0])
    plt.show()


def diagnose_gran(experiment, origItems, gran_name="cluster"):
    gran_df = experiment.gran_experiments["cluster"].annodf
    colors = ["g", "b", "r", "m", "y", "c", "k", "#22aaff", "#ff22aa", "#aaff22"]

    fig, axs = plt.subplots(2, len(origItems), sharex=True, sharey=True, figsize=(4*len(origItems),6))
    
    for col, origItem in enumerate(origItems):
        gran_idf = gran_df[gran_df["origItemID"]==origItem]
        for gvr in experiment.golddict.get(origItem):
            plot_vrimg(gvr, alpha=0.5,  ax=axs[0, col])
        for _, row in gran_idf.iterrows():
            for vr in row[experiment.label_colname]:
                worker = row[experiment.uid_colname] - gran_idf[experiment.uid_colname].min()
                experiment.cluster_plotter(vr, color=colors[worker % len(colors)], alpha=0.5, ax=axs[0, col])
        axs[0, col].set_title("Labels colored by annotator", fontsize=14)

        for gvr in experiment.golddict.get(origItem):
            plot_vrimg(gvr, alpha=0.5,  ax=axs[1, col])
        for _, row in gran_idf.iterrows():
            for vr in row[experiment.label_colname]:
                cluster = row["newItemID"] - gran_idf["newItemID"].min()
                experiment.cluster_plotter(vr, color=colors[cluster % len(colors)], alpha=0.5, text=cluster, ax=axs[1, col])
        axs[1, col].set_title("Labels colored by partition", fontsize=14)
        for ax in axs[:, col]:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
    plt.gca().invert_yaxis()
    plt.plot()

    sad_select_preds = experiment.sad_preds
    cluster_sad_select_preds = experiment.gran_experiments["cluster"].sad_preds
    cluster_sad_merge_preds = experiment.gran_experiments["cluster"].extra_baseline_labels["SAD Merge"]

    sad_select_scores = experiment.score_preds(sad_select_preds)
    cluster_sad_select_scores = experiment.score_preds(cluster_sad_select_preds)
    cluster_sad_merge_scores = experiment.score_preds(cluster_sad_merge_preds)

    fig, axs = plt.subplots(3, len(origItems), sharex=True, sharey=True, figsize=(4*len(origItems),9))
    for col, origItem in enumerate(origItems):
        gran_idf = gran_df[gran_df["origItemID"]==origItem]
        for ax in axs[:,col]:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            for gvr in experiment.golddict.get(origItem):
                plot_vrimg(gvr, alpha=0.5, ax=ax)
            for _, row in gran_idf.iterrows():
                for vr in row[experiment.label_colname]:
                    try:
                        experiment.cluster_plotter(vr, color="k:", alpha=0.3, ax=ax)
                    except: # sorry...
                        experiment.cluster_plotter(vr, color="k", alpha=0.2, ax=ax)

        axs[0, col].set_title(F"IOU score: {np.round(sad_select_scores[origItem], 4)}", fontsize=18)
        for vr in sad_select_preds[origItem]:
            experiment.cluster_plotter(vr, color="m", alpha=1, ax=axs[0, col])

        axs[1, col].set_title(F"IOU score: {np.round(cluster_sad_select_scores[origItem], 4)}", fontsize=18)
        for vr in cluster_sad_select_preds[origItem]:
            experiment.cluster_plotter(vr, color="m", alpha=1, ax=axs[1, col])

        axs[2, col].set_title(F"IOU score: {np.round(cluster_sad_merge_scores[origItem], 4)}", fontsize=18)
        for vr in cluster_sad_merge_preds[origItem]:
            experiment.cluster_plotter(vr, color="m", alpha=1, ax=axs[2, col])
            
    axs[0, 0].set_ylabel("SELECT")
    axs[1, 0].set_ylabel("PSR")
    axs[2, 0].set_ylabel("PDMR")

    fig.tight_layout()
    plt.gca().invert_yaxis()
    plt.plot()

