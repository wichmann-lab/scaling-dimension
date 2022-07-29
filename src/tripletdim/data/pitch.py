import numpy as np
from sklearn.utils import Bunch


def make_pitch_helix(n_octaves=3) -> Bunch:
    """ Generate regular helix as model of pitch perception.

    This model bases on Shepard (1965) who described the perception of
    pitch as a helix, where one rotation corresponds to an octave.
    Tones within an octave are perceived with a circular chroma.
    The same note is perceived with the same chroma in all octaves,
    but with increasing tone height.
    
    This is only one possible model for pitch perception.
    Shepard (1982) describes multiple alternative geometric representations.    

    =================   ==============
    Objects             n_octaves * 12
    Dimensionality                   3
    Features            real, positive
    =================   ==============
    
    Read more in the :ref:`User Guide <pitch_helix>`.
    
    Returns:
        data: Dictionary-like object, with the following attributes.
            
            data: Note coordinates in the helix, shape (n_octaves * 12, 3).
                  The first and second dimension is the circular chroma, 
                  the third the height. 
                  The first tone is always a great C (C2) with chroma (0, 0.5) 
                  and height 0.
            feature_names: The names of the similarity columns/rows.
            frequency: Frequency of the notes.
            scientific: Scientific designation of the notes. 
            helmholz: Helmholz designation of the notes. 
            
    
    >>> data = make_pitch_helix(n_octaves=3)
    >>> data.data.shape
    (36, 3)
    
    
    .. topic:: References
        
        - Shepard, R. N. (1965). Approximation to uniform gradients of generalization by monotone transformations of scale. 
          Stimulus Generalization, 94–110.
        - Shepard, R. N. (1982). Geometrical approximations to the structure of musical pitch. 
          Psychological Review, 89(4), 305–333. https://doi.org/10.1037/0033-295X.89.4.305
    """

    notes, octaves = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'], np.arange(2, 2 + n_octaves)
    repeated_notes, repeated_octaves = np.tile(notes, n_octaves), np.repeat(octaves, 12)
    helmholz_notes = list(notes)
    for o in octaves[octaves >= 3]:
        # helmholz designation describes from octave 3
        # the note and lower case and with a prime per further octave.
        helmholz_notes += [(n[0].lower() + ((o - 3) * "'") + n[1:]) for n in notes]
    scientific_notes = [n[0] + str(o) + n[1:] for n, o in zip(repeated_notes, repeated_octaves)]

    radius = 0.5
    frequency = 65.40639 * 2**(np.arange(n_octaves * 12)/12)
    chroma = np.tile([np.cos(np.arange(12)/6 * np.pi) * radius, np.sin(np.arange(12)/6 * np.pi) * radius], reps=n_octaves).T
    heigth = np.arange(len(chroma))
    embedding = np.c_[chroma, heigth]
    feature_names = ['chroma_1', 'chroma_2', 'height']
    return Bunch(data=embedding,
                 feature_names=feature_names,
                 frequency=frequency,
                 scientific=scientific_notes,
                 helmholz=helmholz_notes)


