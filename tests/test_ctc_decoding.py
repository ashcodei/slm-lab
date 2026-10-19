from collections import defaultdict
from itertools import product
import torch
from slm_lab.character_ctc import collapse,labels
from slm_lab.ctc_decoding import prefix_search


def test_prefix_probability_can_favor_a_character_when_greedy_path_is_blank():
    value=labels('a')[0];probabilities=torch.zeros(3,96)
    probabilities[:,0]=.55;probabilities[:,value]=.45
    assert collapse(probabilities.argmax(1))==''
    assert prefix_search(probabilities.log(),beam_width=32,token_prune=95)=='a'


def test_prefix_search_matches_exhaustive_ctc_alignment_sums():
    a,b=labels('ab');probabilities=torch.zeros(4,96)
    probabilities[:,[0,a,b]]=torch.tensor([[.5,.4,.1],[.2,.7,.1],[.2,.1,.7],[.5,.1,.4]])
    masses=defaultdict(float)
    for path in product([0,a,b],repeat=4):
        mass=1.
        for frame,value in enumerate(path):mass*=float(probabilities[frame,value])
        masses[collapse(path)]+=mass
    expected=max(masses,key=masses.get)
    assert prefix_search(probabilities.log(),beam_width=128,token_prune=95)==expected
