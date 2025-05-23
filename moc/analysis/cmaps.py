import seaborn as sns


def get_recalibrators_cmap(df):
    palette = sns.color_palette()

    cmap = {
        r'\texttt{BASE}': palette[0],
        r'\texttt{LR}': palette[1],
        r'\texttt{HDR-R}': palette[2],
    }
    return cmap, None


def get_cmap(df, cmap):
    names = df.name.unique()
    if cmap is None:
        return dict(zip(names, sns.color_palette(n_colors=len(names)))), None
    if type(cmap) is str:
        cmap, style_map = {
            'recalibrators': get_recalibrators_cmap,
        }[cmap](df)
        return cmap, style_map
    raise ValueError(f'Invalid cmap: {cmap}')
