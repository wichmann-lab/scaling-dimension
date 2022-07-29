from matplotlib import rc


COLUMN_WIDTH = {  # in unit pt
    'beamer-169': 398.3386,
}



def figsize(columnwidth, wf=0.5, hf=None):
    """Parameters:
        - wf [float]:  width fraction in columnwidth units
        - hf [float]:  height fraction in columnwidth units.
                       Set by default to golden ratio.
        - columnwidth [float]: width of the column in latex. Get this from LaTeX 
                               using \show\columnwidth
    Returns:  [fig_width,fig_height]: that should be given to matplotlib
    """
    if columnwidth in COLUMN_WIDTH:
        columnwidth = COLUMN_WIDTH[columnwidth]
    if hf is None:
        hf = (5**0.5 - 1) / 2 
    fig_width_pt = columnwidth*wf 
    inches_per_pt = 1 / 72.27 # pt to inch
    fig_width = fig_width_pt * inches_per_pt  # width in inch
    fig_height = fig_width * hf      # height in inches
    return [fig_width, fig_height]

    
def tex_escape(s: str) -> str:
    escape_seq = ['#', '$', '%', '&', '~', '_', '^', '{', '}', '\\(', '\\)', '\\[', '\\]']
    for seq in escape_seq:
        loc = 0
        while True:
            loc = s.find(seq, loc)
            if loc == -1:
                break
            s = s[:loc] + '\\' + s[loc:]
            loc += 1 + len(seq)
    return s